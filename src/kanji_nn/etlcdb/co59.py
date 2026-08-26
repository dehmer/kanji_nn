from pathlib import Path


_EUC_CO59 = Path(__file__).with_name("euc_co59.dat")


class CO59_to_unicode:
    def __init__(self, filename):
        with open(filename, 'r', encoding = 'euc-jp') as f:
            co59t = f.read()
        co59l = co59t.split()
        self.conv = {}
        for c in co59l:
            ch = c.split(':')
            co = ch[1].split(',')
            co59c = (int(co[0]), int(co[1]))
            self.conv[co59c] = ch[0]

    def __call__(self, co59):
        return self.conv[co59]


co59_to_unicode = CO59_to_unicode(_EUC_CO59)
