{
  description = "Python 3.9 development environment";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
  in {
    devShells.${system}.default = pkgs.mkShellNoCC {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

      buildInputs = with pkgs; [
        ffmpeg

        python311
        python311Packages.pip
        python311Packages.virtualenv
      ];
    };
  };
}
