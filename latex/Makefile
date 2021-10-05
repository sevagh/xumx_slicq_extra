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

poster-dev: export LATEXMK_FLAGS="-pvc"
poster-dev: poster
poster: tex_fname="poster.tex"
poster: texpdf

chapter-dev: export LATEXMK_FLAGS="-pvc"
chapter-dev: chapter
chapter: tex_fname="report_${CHAPTER}.tex"
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
	-rm *.log
