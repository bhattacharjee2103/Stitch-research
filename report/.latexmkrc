add_cus_dep('nlo', 'nls', 0, 'makenomenclature');
sub makenomenclature {
  return system("makeindex -s nomencl.ist -o \"$_[0].nls\" \"$_[0].nlo\"");
}
$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
