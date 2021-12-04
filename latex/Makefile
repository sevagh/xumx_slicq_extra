LATEXMK_FLAGS ?= ""
CHAPTER ?= "ch0"

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 

report-dev: export LATEXMK_FLAGS="-pvc"
report-dev: report
report: tex_fname="report.tex"
report: texpdf

proposal-dev: export LATEXMK_FLAGS="-pvc"
proposal-dev: proposal
proposal: tex_fname="proposal.tex"
proposal: texpdf

mml21paper-dev: export LATEXMK_FLAGS="-pvc"
mml21paper-dev: mml21paper
mml21paper: tex_fname="mml21_paper.tex"
mml21paper: texpdf

mml21presentation-dev: export LATEXMK_FLAGS="-pvc"
mml21presentation-dev: mml21presentation
mml21presentation: tex_fname="mml21_presentation.tex"
mml21presentation: texpdf

all: report

clean:
	latexmk -C
	-rm *.lol
	-rm *.bbl
	-rm *.run.xml
	-rm -r _minted*
	-rm *.nav
	-rm *.pyg
	-rm *.snm
	-rm *.vrb
	-rm *.fls
	-rm *.toc
	-rm *.log
