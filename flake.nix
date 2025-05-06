{
  description = "Arti project flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # Use unstable channel
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        pythonEnv = pkgs.python313.withPackages (
          ps: with ps; [
            numpy
            polars
            pytorch
            matplotlib
            jupyter
            scikit-learn
            requests
            # Add other Python packages here
          ]
        );
      in
      {
        # Set up environment variables if needed
        # Development shell (for `nix develop`)
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            # Add other build inputs here
          ];

          packages = with pkgs; [
            # Add other development dependencies here
            python313
          ];
          shellHook = ''
            export PYTHONPATH=${pythonEnv}/${pythonEnv.sitePackages}
          '';

        };
      }
    );
}
