import numpy as np

Office = [0. for i in range(6)]
o_i = 0

OfficeHome = [0. for j in range(12)]
of_i = 0

nbr = 1

with open(f"log.txt", "r", encoding="utf-8") as f:
    res = f.readlines()
    for i in res:
        if len(i) > 40:
            s = i.split(' ')
            if s[1] == "Office":
                Office[o_i] = float(s[6][:-1])
                o_i += 1
                if o_i == 6:
                    o_i = 0
                    avg = np.mean(Office)
                    with open("result.txt", "a+") as wf:
                        wf.write(f"OPDA Adaptation ON Office, nbr={nbr}\n")
                        wf.write(f"OPDA Office -3 Avg.: H-Score: {avg:.3f}\n")

            elif s[1] == "OfficeHome":
                OfficeHome[of_i] = float(s[6][:-1])
                of_i += 1
                if of_i == 12:
                    of_i = 0
                    avg = np.mean(OfficeHome)
                    with open("result.txt", "a+") as wf:
                        wf.write(f"OPDA Adaptation ON Office-Home, nbr={nbr}\n")
                        wf.write(f"OPDA Office-Home -3 Avg.: H-Score: {avg:.3f}\n")
                    nbr += 1
