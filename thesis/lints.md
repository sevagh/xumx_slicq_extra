# replacing plural we/our/us with singular i/me

```
$ rg -i '(.*\W+we\W+.*|.*\W+us\W+.*|.*\W+our\W+.*)' ch5.tex
```

# acronyms

```
$ rg -o '\W[A-Z]{3,}\W' -g 'ch*.tex'
```

# todos

```
sevagh:latex $ rg 'todo|\\hl\{|explanation' -g 'ch*.tex'
ch6.tex
82:\todo[inline]{fix up the wording}
125:\todo[inline]{actually do this, talk about ISMIR2021 challenge, submission repo, experiments, etc.}

ch2.tex
829:\todo[inline]{repeat/distinguish the differences between music source separation and music demixing}
861:\explanation{why is 14 tracks significant? because you told me to do that}
866:\todo[inline]{this section needs an intro, why do we need to evaluate?}
899:\todo[inline]{blurb on casa and bss}
1009:\todo[inline]{this section needs an intro}
1075:\todo[inline]{better intro, why are we talking about this}
```

# too many consecutive spaces

```
$ rg '[^\s]([ ]{2,})[^\s]' -g 'ch*.tex'
```

# fixing captions without trailing periods

```
rg '\\caption\{.*[^\.]\}$'
rg '\\subfloat\[.*[^\.]\]\{\\includegraphics'
```
