from io import TextIOWrapper
import time

def read_metadata(f: TextIOWrapper):
    meta = {}
    while (line := f.readline()).strip() != "<END OF METADATA>":
        k, v = line.split(">", 1)
        meta[k[1:]] = v.strip()
    return meta

def read_network(file: str):
    res = []
    with open(file) as f:
        meta = read_metadata(f)
        while (headline := f.readline()).strip() == "":
            pass
        columns = headline[1:].split()
        init_column = columns.index("init_node")
        term_column = columns.index("term_node")
        cap_column = columns.index("capacity")
        time_column = columns.index("free_flow_time")
        for line in f:
            line = line.strip()
            if line != "":
                vals = line[:-1].split()
                res.append((int(vals[init_column]), int(vals[term_column]), 
                            float(vals[cap_column]), float(vals[time_column])))
        return int(meta["NUMBER OF NODES"]), res
    
def read_trips(file: str):
    res = []
    with open(file) as f:
        meta = read_metadata(f)
        origin = -1
        for line in f:
            if line.strip() != "":
                if line.startswith("Origin"):
                    origin = int(line.split()[1])
                else:
                    parts = line.split()
                    for i in range(len(parts) // 3):
                        res.append((origin, int(parts[3*i]), float(parts[3*i+2][:-1])))
    return res
