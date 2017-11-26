import xml.etree.ElementTree as et
import numpy as np

branch_insts = ["ret", "br", "switch", "indirectbr", "invoke", "resume",
                "unreachable", "cleanupret", "catchret", "catchpad",
                "catchswitch"]

binary_insts = ["add", "fadd", "sub", "fsub", "mul", "fmul", "udiv",
                "sdiv", "fdiv", "urem", "srem", "frem"]

logical_insts = ["and", "or", "xor"]

memory_insts = ["alloca", "load", "store", "cmpxchg", "atomicrmw", "fence",
                "getelementptr"]

convert_insts = ["trunc", "zext", "sext", "fptrunc", "fpext", "fptoui",
                 "fptosi", "uitofp", "sitofp", "inttoptr", "ptrtoint",
                 "bitcast", "addrspacecast"]

other_insts = ["icmp", "fcmp", "phi", "select", "call", "shl", "lshr",
               "ashr", "va_arg", "extractelement", "insertelement",
               "shufflevector", "extractvalue", "insertvalue",
               "landingpad", "cleanuppad"]

all_insts = branch_insts + binary_insts + logical_insts
all_insts += memory_insts + convert_insts + other_insts

inst_types = [branch_insts, binary_insts, logical_insts,
              memory_insts, convert_insts, other_insts]

bb_vector_len = len(all_insts) + len(inst_types)
path_bb_len = 20

hot_path_tensors = np.zeros((12500, path_bb_len, bb_vector_len))
cold_path_tensors = np.zeros((12500, path_bb_len, bb_vector_len))
batch_nums = [0, 0]


def get_inst_num(inst):
    return all_insts.index(inst)


def get_inst_type_num(inst):
    for i, inst_type in enumerate(inst_types):
        if inst in inst_type:
            return i + len(all_insts)


def read_basic_block(bb):
    bb_vec = np.zeros(bb_vector_len)
    for opcode in bb.findall('opcode'):
        bb_vec[get_inst_num(opcode.text)] += 1
        bb_vec[get_inst_type_num(opcode.text)] += 1
    return bb_vec


def read_path(path):
    path_tensor = np.zeros((path_bb_len, bb_vector_len))
    for i, bb in enumerate(path.findall('basic_block')):
        path_tensor[i] = read_basic_block(bb)
    frequency = float(path.findall('frequency')[0].text)
    return path_tensor, frequency


def read_data(file_name):
    e = et.parse(file_name)
    for path in e.findall('path'):
        path_tensor, frequency = read_path(path)
        if(frequency > .5 and batch_nums[0] < 12500):
            hot_path_tensors[batch_nums[0]] = path_tensor
            batch_nums[0] += 1
        elif(frequency < .5 and batch_nums[1] < 12500):
            cold_path_tensors[batch_nums[1]] = path_tensor
            batch_nums[1] += 1

read_data_arr = ["583sample.txt"]
for file_name in read_data_arr:
    read_data(file_name)

all_data = np.append(cold_path_tensors, hot_path_tensors, axis=0)
np.save("data.npy", all_data)

for path in hot_path_tensors + cold_path_tensors:
    np.random.shuffle(path)
all_data = np.append(cold_path_tensors, hot_path_tensors, axis=0)
np.save("shuffled_data.npy", all_data)