import re


strs = [
"block 0, f 0-0: torch.Size([6, 2, 1, 28]), torch.Size([2, 1, 98]), torch.Size([2, 1, 168])",
"block 1, f 1-86: torch.Size([6, 2, 86, 16]), torch.Size([2, 86, 56]), torch.Size([2, 86, 96])",
"block 2, f 87-100: torch.Size([6, 2, 14, 20]), torch.Size([2, 14, 70]), torch.Size([2, 14, 120])",
"block 3, f 101-111: torch.Size([6, 2, 11, 24]), torch.Size([2, 11, 84]), torch.Size([2, 11, 144])",
"block 4, f 112-120: torch.Size([6, 2, 9, 28]), torch.Size([2, 9, 98]), torch.Size([2, 9, 168])",
"block 5, f 121-128: torch.Size([6, 2, 8, 32]), torch.Size([2, 8, 112]), torch.Size([2, 8, 192])",
"block 6, f 129-135: torch.Size([6, 2, 7, 36]), torch.Size([2, 7, 126]), torch.Size([2, 7, 216])",
"block 7, f 136-141: torch.Size([6, 2, 6, 40]), torch.Size([2, 6, 140]), torch.Size([2, 6, 240])",
"block 8, f 142-147: torch.Size([6, 2, 6, 44]), torch.Size([2, 6, 154]), torch.Size([2, 6, 264])",
"block 9, f 148-152: torch.Size([6, 2, 5, 48]), torch.Size([2, 5, 168]), torch.Size([2, 5, 288])",
"block 10, f 153-157: torch.Size([6, 2, 5, 52]), torch.Size([2, 5, 182]), torch.Size([2, 5, 312])",
"block 11, f 158-161: torch.Size([6, 2, 4, 56]), torch.Size([2, 4, 196]), torch.Size([2, 4, 336])",
"block 12, f 162-166: torch.Size([6, 2, 5, 60]), torch.Size([2, 5, 210]), torch.Size([2, 5, 360])",
"block 13, f 167-169: torch.Size([6, 2, 3, 64]), torch.Size([2, 3, 224]), torch.Size([2, 3, 384])",
"block 14, f 170-173: torch.Size([6, 2, 4, 68]), torch.Size([2, 4, 238]), torch.Size([2, 4, 408])",
"block 15, f 174-177: torch.Size([6, 2, 4, 72]), torch.Size([2, 4, 252]), torch.Size([2, 4, 432])",
"block 16, f 178-180: torch.Size([6, 2, 3, 76]), torch.Size([2, 3, 266]), torch.Size([2, 3, 456])",
"block 17, f 181-183: torch.Size([6, 2, 3, 80]), torch.Size([2, 3, 280]), torch.Size([2, 3, 480])",
"block 18, f 184-186: torch.Size([6, 2, 3, 84]), torch.Size([2, 3, 294]), torch.Size([2, 3, 504])",
"block 19, f 187-189: torch.Size([6, 2, 3, 88]), torch.Size([2, 3, 308]), torch.Size([2, 3, 528])",
"block 20, f 190-191: torch.Size([6, 2, 2, 92]), torch.Size([2, 2, 322]), torch.Size([2, 2, 552])",
"block 21, f 192-194: torch.Size([6, 2, 3, 96]), torch.Size([2, 3, 336]), torch.Size([2, 3, 576])",
"block 22, f 195-196: torch.Size([6, 2, 2, 100]), torch.Size([2, 2, 350]), torch.Size([2, 2, 600])",
"block 23, f 197-199: torch.Size([6, 2, 3, 104]), torch.Size([2, 3, 364]), torch.Size([2, 3, 624])",
"block 24, f 200-201: torch.Size([6, 2, 2, 108]), torch.Size([2, 2, 378]), torch.Size([2, 2, 648])",
"block 25, f 202-203: torch.Size([6, 2, 2, 112]), torch.Size([2, 2, 392]), torch.Size([2, 2, 672])",
"block 26, f 204-205: torch.Size([6, 2, 2, 116]), torch.Size([2, 2, 406]), torch.Size([2, 2, 696])",
"block 27, f 206-207: torch.Size([6, 2, 2, 120]), torch.Size([2, 2, 420]), torch.Size([2, 2, 720])",
"block 28, f 208-209: torch.Size([6, 2, 2, 124]), torch.Size([2, 2, 434]), torch.Size([2, 2, 744])",
"block 29, f 210-211: torch.Size([6, 2, 2, 128]), torch.Size([2, 2, 448]), torch.Size([2, 2, 768])",
"block 30, f 212-213: torch.Size([6, 2, 2, 132]), torch.Size([2, 2, 462]), torch.Size([2, 2, 792])",
"block 31, f 214-215: torch.Size([6, 2, 2, 136]), torch.Size([2, 2, 476]), torch.Size([2, 2, 816])",
"block 32, f 216-217: torch.Size([6, 2, 2, 140]), torch.Size([2, 2, 490]), torch.Size([2, 2, 840])",
"block 33, f 218-218: torch.Size([6, 2, 1, 144]), torch.Size([2, 1, 504]), torch.Size([2, 1, 864])",
"block 34, f 219-220: torch.Size([6, 2, 2, 148]), torch.Size([2, 2, 518]), torch.Size([2, 2, 888])",
"block 35, f 221-222: torch.Size([6, 2, 2, 152]), torch.Size([2, 2, 532]), torch.Size([2, 2, 912])",
"block 36, f 223-223: torch.Size([6, 2, 1, 156]), torch.Size([2, 1, 546]), torch.Size([2, 1, 936])",
"block 37, f 224-225: torch.Size([6, 2, 2, 160]), torch.Size([2, 2, 560]), torch.Size([2, 2, 960])",
"block 38, f 226-226: torch.Size([6, 2, 1, 164]), torch.Size([2, 1, 574]), torch.Size([2, 1, 984])",
"block 39, f 227-228: torch.Size([6, 2, 2, 168]), torch.Size([2, 2, 588]), torch.Size([2, 2, 1008])",
"block 40, f 229-229: torch.Size([6, 2, 1, 172]), torch.Size([2, 1, 602]), torch.Size([2, 1, 1032])",
"block 41, f 230-231: torch.Size([6, 2, 2, 176]), torch.Size([2, 2, 616]), torch.Size([2, 2, 1056])",
"block 42, f 232-232: torch.Size([6, 2, 1, 180]), torch.Size([2, 1, 630]), torch.Size([2, 1, 1080])",
"block 43, f 233-233: torch.Size([6, 2, 1, 184]), torch.Size([2, 1, 644]), torch.Size([2, 1, 1104])",
"block 44, f 234-235: torch.Size([6, 2, 2, 188]), torch.Size([2, 2, 658]), torch.Size([2, 2, 1128])",
"block 45, f 236-236: torch.Size([6, 2, 1, 192]), torch.Size([2, 1, 672]), torch.Size([2, 1, 1152])",
"block 46, f 237-237: torch.Size([6, 2, 1, 196]), torch.Size([2, 1, 686]), torch.Size([2, 1, 1176])",
"block 47, f 238-238: torch.Size([6, 2, 1, 200]), torch.Size([2, 1, 700]), torch.Size([2, 1, 1200])",
"block 48, f 239-240: torch.Size([6, 2, 2, 204]), torch.Size([2, 2, 714]), torch.Size([2, 2, 1224])",
"block 49, f 241-241: torch.Size([6, 2, 1, 208]), torch.Size([2, 1, 728]), torch.Size([2, 1, 1248])",
"block 50, f 242-242: torch.Size([6, 2, 1, 212]), torch.Size([2, 1, 742]), torch.Size([2, 1, 1272])",
"block 51, f 243-243: torch.Size([6, 2, 1, 216]), torch.Size([2, 1, 756]), torch.Size([2, 1, 1296])",
"block 52, f 244-244: torch.Size([6, 2, 1, 220]), torch.Size([2, 1, 770]), torch.Size([2, 1, 1320])",
"block 53, f 245-245: torch.Size([6, 2, 1, 224]), torch.Size([2, 1, 784]), torch.Size([2, 1, 1344])",
"block 54, f 246-246: torch.Size([6, 2, 1, 228]), torch.Size([2, 1, 798]), torch.Size([2, 1, 1368])",
"block 55, f 247-248: torch.Size([6, 2, 2, 232]), torch.Size([2, 2, 812]), torch.Size([2, 2, 1392])",
"block 56, f 249-249: torch.Size([6, 2, 1, 236]), torch.Size([2, 1, 826]), torch.Size([2, 1, 1416])",
"block 57, f 250-250: torch.Size([6, 2, 1, 240]), torch.Size([2, 1, 840]), torch.Size([2, 1, 1440])",
"block 58, f 251-251: torch.Size([6, 2, 1, 244]), torch.Size([2, 1, 854]), torch.Size([2, 1, 1464])",
"block 59, f 252-252: torch.Size([6, 2, 1, 248]), torch.Size([2, 1, 868]), torch.Size([2, 1, 1488])",
"block 60, f 253-253: torch.Size([6, 2, 1, 252]), torch.Size([2, 1, 882]), torch.Size([2, 1, 1512])",
"block 61, f 254-254: torch.Size([6, 2, 1, 256]), torch.Size([2, 1, 896]), torch.Size([2, 1, 1536])",
"block 62, f 255-255: torch.Size([6, 2, 1, 264]), torch.Size([2, 1, 924]), torch.Size([2, 1, 1584])",
"block 63, f 256-256: torch.Size([6, 2, 1, 268]), torch.Size([2, 1, 938]), torch.Size([2, 1, 1608])",
"block 64, f 257-257: torch.Size([6, 2, 1, 272]), torch.Size([2, 1, 952]), torch.Size([2, 1, 1632])",
"block 65, f 258-258: torch.Size([6, 2, 1, 276]), torch.Size([2, 1, 966]), torch.Size([2, 1, 1656])",
"block 66, f 259-259: torch.Size([6, 2, 1, 280]), torch.Size([2, 1, 980]), torch.Size([2, 1, 1680])",
"block 67, f 260-260: torch.Size([6, 2, 1, 284]), torch.Size([2, 1, 994]), torch.Size([2, 1, 1704])",
"block 68, f 261-261: torch.Size([6, 2, 1, 288]), torch.Size([2, 1, 1008]), torch.Size([2, 1, 1728])",
"block 69, f 262-262: torch.Size([6, 2, 1, 292]), torch.Size([2, 1, 1022]), torch.Size([2, 1, 1752])",
]

if __name__ == '__main__':
    for string in strs:
        block = string.split(',')[0].split()[-1]
        f = '--'.join(string.split(',')[1].split(':')[0].split()[-1].split('-'))

        fmin, fmax = f.split('--')
        fs = int(fmax) - int(fmin)+1

        if fmax==fmin:
            f = fmax

        parens = re.findall(r'\((.*?)\)', string)

        t_3d = parens[0].split(',')[-1][:-1]
        t_2d_ola = parens[1].split(',')[-1][:-1]
        t_2d_flat = parens[2].split(',')[-1][:-1]

        #print(f'{float(t_2d_ola):,.0f}')
        #print(f'{t_2d_flat}')

        spc = ' \ '

        print(f'{block} & {f} & (9,{spc}{fs},{spc}{float(t_3d):,.0f}) & ({fs},{spc}{float(t_2d_ola):,.0f}) & ({fs},{spc}{float(t_2d_flat):,.0f}) \\\\')
        print('\hline')
