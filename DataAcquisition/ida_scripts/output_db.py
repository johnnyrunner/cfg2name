import idc
import sys
idc.gen_file(idc.GENFLG_MAPNAME, sys.argv[1], 0, idc.BADADDR, 0)
import ida_pro
ida_pro.qexit()