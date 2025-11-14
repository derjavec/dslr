{
  description = "Environnement de développement Python pour l'analyse de données";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:

    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (ps: [
          ps.pandas
          ps.numpy
          ps.scikit-learn


          ps.matplotlib
          ps.seaborn
          ps.statsmodels
        ]);

      in
      {

        devShells.default = pkgs.mkShell {

          buildInputs = [
            pythonEnv
          ];
        };
      }
    );
}
