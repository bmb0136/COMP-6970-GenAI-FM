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
      let
        torch = pp.torch.override { inherit cudaSupport; };
      in
      with pp;
      [
        numpy
        matplotlib
        pandas
        openpyxl
        jupyter
        ipympl
        torch
        datasets
        requests
        aiohttp
        sentencepiece
        (transformers.override { inherit torch; })
      ]
    ))
  ];
}
