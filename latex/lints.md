# replacing plural we/our/us with singular i/me

```
$ rg -i '(.*\W+we\W+.*|.*\W+us\W+.*|.*\W+our\W+.*)' ch5.tex
```

# acronyms

```
$ rg -o '\W[A-Z]{3,}\W' -g 'ch*.tex'
```
