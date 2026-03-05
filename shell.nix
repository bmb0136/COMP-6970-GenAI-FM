{
  pkgs ? import <nixpkgs>,
  mkShell ? pkgs.mkShell,
  cudaSupport ? false,
  ...
}:
mkShell {
  packages = [
    (pkgs.python3.withPackages (
      pp:
      with pp;
      [
        numpy
        matplotlib
        pandas
        openpyxl
        jupyter
        ipympl
        (torch.override { inherit cudaSupport; })
        datasets
      ]
    ))
  ];
}
