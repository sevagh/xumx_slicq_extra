LATEXMK_FLAGS := "-jobname=mdx21_hanssian"

texpdf:
	latexmk $(tex_fname) ${LATEXMK_FLAGS} 

arxiv-dev: export LATEXMK_FLAGS="-pvc"
arxiv-dev: arxiv
arxiv: tex_fname="paper.tex"
arxiv: texpdf

all: arxiv

clean:
	latexmk -C
	-rm *.lol
	-rm *.run.xml
	-rm -r _minted*
	-rm *.nav
	-rm *.pyg
	-rm *.snm
	-rm *.vrb
	-rm *.fls
	-rm *.toc
	-rm *.log
