import struct, random

def write_bin(path, count, dim, seed):
    random.seed(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<II", count, dim))
        for _ in range(count * dim):
            f.write(struct.pack("<f", random.random()))

write_bin("data.bin", 2000, 32, 1)
write_bin("queries.bin", 5, 32, 2)