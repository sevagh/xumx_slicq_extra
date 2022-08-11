LATEXMK_FLAGS ?= ""

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 

report-dev: export LATEXMK_FLAGS="-pvc"
report-dev: report
report: tex_fname="report.tex"
report: texpdf

all: report

ghostscript-compressed-thesis: report
ghostscript-compressed-thesis:
	gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dPDFSETTINGS=/printer -dNOPAUSE -dQUIET -dBATCH -sOutputFile=thesis.pdf report.pdf

ghostscript-final-thesis-pdfa: report
ghostscript-final-thesis-pdfa:
	gs -sDEVICE=pdfwrite -dPDFA -dBATCH -dNOPAUSE -sColorConversionStrategy=UseDeviceIndependentColor -dPDFACompatibilityPolicy=2 -sOutputFile=thesis.pdf report.pdf

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
