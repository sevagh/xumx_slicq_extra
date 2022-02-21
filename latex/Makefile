LATEXMK_FLAGS ?= ""

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 

report-dev: export LATEXMK_FLAGS="-pvc"
report-dev: report
report: tex_fname="report.tex"
report: texpdf

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
