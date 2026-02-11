{
  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      perSystem =
        { pkgs, ... }:
        {
          devShells.default = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (
                pp: with pp; [
                  numpy
                  matplotlib
                  pandas
                  openpyxl
                  jupyter
                  ipympl
                  (torch.override { cudaSupport = true; })
                ]
              ))
            ];
          };
        };
    };
}
