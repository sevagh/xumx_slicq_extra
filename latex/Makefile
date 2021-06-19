dev-report:
	latexmk -pvc report.tex

dev-proposal:
	latexmk -pvc proposal.tex

dev-presentation01:
	latexmk -pvc presentation01.tex

dev-presentation02:
	latexmk -pvc presentation02.tex

all: clean report proposal presentation01 presentation02

report:
	latexmk report.tex

proposal:
	latexmk proposal.tex

presentation01:
	latexmk presentation01.tex

presentation01:
	latexmk presentation02.tex

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
