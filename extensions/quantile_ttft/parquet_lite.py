"""Minimal pure-Python Parquet reader (flat schema, INT64/DOUBLE/INT32, SNAPPY or UNCOMPRESSED,
PLAIN / PLAIN_DICTIONARY / RLE_DICTIONARY, data page v1/v2, required or optional columns).
Written because pyarrow is unavailable in this sandbox. Usage: python parquet_lite.py in.parquet out.csv
"""
import struct
import sys

import numpy as np


# ---------------- thrift compact protocol
class TC:
    def __init__(self, b, pos=0):
        self.b, self.p = b, pos

    def varint(self):
        r = s = 0
        while True:
            x = self.b[self.p]; self.p += 1
            r |= (x & 0x7F) << s
            if not x & 0x80:
                return r
            s += 7

    def zz(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)

    def read_val(self, t):
        if t in (1, 2):
            return t == 1
        if t == 3:
            v = self.b[self.p]; self.p += 1; return v
        if t in (4, 5, 6):
            return self.zz()
        if t == 7:
            v = struct.unpack_from("<d", self.b, self.p)[0]; self.p += 8; return v
        if t == 8:
            n = self.varint(); v = bytes(self.b[self.p:self.p + n]); self.p += n; return v
        if t in (9, 10):
            h = self.b[self.p]; self.p += 1
            n, et = h >> 4, h & 0xF
            if n == 15:
                n = self.varint()
            return [self.read_val(et) for _ in range(n)]
        if t == 11:
            n = self.varint()
            if n:
                kv = self.b[self.p]; self.p += 1
                return [(self.read_val(kv >> 4), self.read_val(kv & 0xF)) for _ in range(n)]
            return []
        if t == 12:
            return self.struct()
        raise ValueError(f"thrift type {t}")

    def struct(self):
        d, last = {}, 0
        while True:
            h = self.b[self.p]; self.p += 1
            if h == 0:
                return d
            t, delta = h & 0xF, h >> 4
            fid = last + delta if delta else self.zz()
            last = fid
            d[fid] = self.read_val(t)


# ---------------- snappy
def snappy_decompress(b):
    tc = TC(b)
    n = tc.varint()
    p = tc.p
    out = bytearray()
    while p < len(b):
        tag = b[p]; p += 1
        k = tag & 3
        if k == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(b[p:p + nb], "little"); p += nb
            ln += 1
            out += b[p:p + ln]; p += ln
            continue
        if k == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | b[p]; p += 1
        elif k == 2:
            ln = (tag >> 2) + 1
            off = int.from_bytes(b[p:p + 2], "little"); p += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(b[p:p + 4], "little"); p += 4
        s = len(out) - off
        for i in range(ln):           # may overlap
            out.append(out[s + i])
    assert len(out) == n, (len(out), n)
    return bytes(out)


# ---------------- RLE / bit-packed hybrid
def rle_hybrid(b, bw, count):
    out, p = [], 0
    bytew = (bw + 7) // 8
    while len(out) < count and p < len(b):
        tc = TC(b, p)
        h = tc.varint(); p = tc.p
        if h & 1:
            groups = h >> 1
            nbytes = groups * bw
            chunk = int.from_bytes(b[p:p + nbytes], "little"); p += nbytes
            mask = (1 << bw) - 1
            for i in range(groups * 8):
                out.append((chunk >> (i * bw)) & mask if bw else 0)
        else:
            run = h >> 1
            v = int.from_bytes(b[p:p + bytew], "little") if bytew else 0; p += bytew
            out.extend([v] * run)
    return out[:count]


NP = {1: "<i4", 2: "<i8", 4: "<f4", 5: "<f8"}


def read_parquet(path):
    raw = open(path, "rb").read()
    assert raw[:4] == b"PAR1" and raw[-4:] == b"PAR1"
    flen = struct.unpack("<I", raw[-8:-4])[0]
    meta = TC(raw, len(raw) - 8 - flen).struct()
    schema = meta[2]
    leaves = [s for s in schema[1:]]
    names = [s[4].decode() for s in leaves]
    cols = {n: [] for n in names}
    for rg in meta[4]:
        for ci, cc in enumerate(rg[1]):
            md = cc[3]
            ptype, codec, nvals = md[1], md[4], md[5]
            optional = leaves[ci].get(3, 0) == 1
            start = md.get(11, md[9])
            end = md[9] + md[7] if 11 not in md else md[11] + md[7]
            start = md[11] if 11 in md else md[9]
            p = start
            dictionary = None
            got = []
            while len(got) < nvals:
                tc = TC(raw, p)
                ph = tc.struct()
                p = tc.p
                body = raw[p:p + ph[3]]; p += ph[3]
                ptype_page = ph[1]
                if ptype_page == 2:          # dictionary page
                    data = snappy_decompress(body) if codec == 1 else body
                    dictionary = np.frombuffer(data, NP[ptype], ph[7][1])
                    continue
                if ptype_page == 0:          # data page v1
                    h = ph[5]
                    n, enc = h[1], h[2]
                    data = snappy_decompress(body) if codec == 1 else body
                    q = 0
                    if optional:
                        ln = struct.unpack_from("<I", data, 0)[0]
                        defs = rle_hybrid(data[4:4 + ln], 1, n); q = 4 + ln
                    else:
                        defs = [1] * n
                elif ptype_page == 3:        # data page v2
                    h = ph[8]
                    n, enc = h[1], h[4]
                    dl, rl = h[5], h[6]
                    lv = body[:dl + rl]
                    rest = body[dl + rl:]
                    if h.get(7, True) and codec == 1:
                        rest = snappy_decompress(rest)
                    defs = rle_hybrid(lv[rl:rl + dl], 1, n) if optional else [1] * n
                    data, q = rest, 0
                else:
                    raise ValueError(f"page type {ptype_page}")
                nn = sum(defs)
                if enc == 0:
                    vals = np.frombuffer(data, NP[ptype], nn, q)
                elif enc in (2, 8):
                    bw = data[q]
                    idx = rle_hybrid(data[q + 1:], bw, nn)
                    vals = dictionary[np.array(idx, dtype=np.int64)]
                else:
                    raise ValueError(f"encoding {enc}")
                it = iter(vals)
                got.extend(next(it) if d else np.nan for d in defs)
            cols[names[ci]].extend(got)
    return {n: np.array(v) for n, v in cols.items()}


if __name__ == "__main__":
    import pandas as pd
    d = read_parquet(sys.argv[1])
    pd.DataFrame(d).to_csv(sys.argv[2], index=False)
    print({k: (v.dtype, len(v)) for k, v in d.items()})
