{
  description = "Python 3.9 development environment";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        pkgs.portaudio
        pkgs.python311
        pkgs.python311Packages.pip
        pkgs.python311Packages.virtualenv
      ];
    };
  };
}
