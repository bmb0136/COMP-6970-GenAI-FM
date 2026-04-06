{
  pkgs ? import <nixpkgs>,
  mkShell ? pkgs.mkShell,
  cudaSupport ? false,
  ...
}:
mkShell {
  packages = [
    pkgs.wget
    pkgs.protobuf
    (pkgs.python3.withPackages (
      pp:
      let
        torch = pp.torch.override { inherit cudaSupport; };
        accelerate = pp.accelerate.override { inherit torch; };
        transformers = pp.transformers.override { inherit torch; };
        trl = pp.trl.override { inherit accelerate transformers; };
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
        protobuf
        wget
        trl
        transformers
        accelerate
      ]
    ))
  ];
}
