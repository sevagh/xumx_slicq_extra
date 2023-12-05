LATEXMK_FLAGS ?= ""

all: presentation handout

# handout copy: no notes
handout:
	latexmk -jobname=handout ${LATEXMK_FLAGS} presentation.tex

# presenter copy: full size with notes
presentation-dev-pympress: export LATEXMK_FLAGS=-pvc -e '$$pdf_previewer=q[start pympress];'
presentation-dev-pympress: presentation
presentation-dev: export LATEXMK_FLAGS=-pvc
presentation-dev: presentation
presentation:
	latexmk -jobname=presentation ${LATEXMK_FLAGS} presentation.tex

clean:
	latexmk -C
	latexmk -jobname=handout -C
	latexmk -jobname=presentation -C
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

.PHONY: clean
