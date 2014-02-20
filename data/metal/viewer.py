import array as ar
import pylab as pl

with open("double_subject.bin", "rb") as f:

    S = ar.array("d", f.read())

pl.rcParams.update({'font.size': 28})

ax2=pl.subplot(111)
ax2.set_title("$S$")
ax2.plot(S, c="black", linewidth=2)
ax2.set_xticks([])
ax2.set_yticks([])

pl.tight_layout()
pl.show()
