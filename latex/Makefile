LATEXMK_FLAGS ?= ""
CHAPTER ?= "ch0"

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 

report-refs-dev: export LATEXMK_FLAGS="-pvc"
report-refs-dev: report-refs
report-refs: tex_fname="report_refs.tex"
report-refs: texpdf

report-dev: export LATEXMK_FLAGS="-pvc"
report-dev: report
report: tex_fname="report.tex"
report: texpdf

proposal-dev: export LATEXMK_FLAGS="-pvc"
proposal-dev: proposal
proposal: tex_fname="proposal.tex"
proposal: texpdf

chapter-dev: export LATEXMK_FLAGS="-pvc"
chapter-dev: chapter
chapter: tex_fname="${CHAPTER}.tex"
chapter: texpdf

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
