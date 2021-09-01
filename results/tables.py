def parse_results(lines):
    all_results = {}
    for c in lines:
        result = c.split(': ')
        szr_type = result[0]
        szr_type = ' '.join(szr_type.split(','))
        num = float(result[1])
        num = "%.3f" % num
        all_results[szr_type] = num
    return all_results

def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    lines = [x.strip() for x in content]
    return parse_results(lines)


acc = read_file("szr_ACC_single_wrst.txt")
eda = read_file("szr_EDA_single_wrst.txt")
bvp = read_file("szr_BVP_single_wrst.txt")
acc_eda = read_file("szr_ACC_EDA_single_wrst.txt")
acc_bvp = read_file("szr_ACC_BVP_single_wrst.txt")
eda_bvp = read_file("szr_EDA_BVP_single_wrst.txt")
acc_bvp_eda = read_file("szr_ACC_BVP_EDA_single_wrst.txt")


for k in sorted(acc.keys()):
    print(k+','+acc[k]+','+eda[k]+','+bvp[k]+','+acc_bvp[k]+','+eda_bvp[k]+','+acc_eda[k]+','+acc_bvp_eda[k])