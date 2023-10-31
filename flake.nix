{
  description = "UB-NSSD development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = false;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      buildInputs = with pkgs; let
        tensorflow-hub = python311Packages.buildPythonPackage rec {
          pname = "tensorflow_hub";
          version = "0.15.0";
          format = "wheel";

          src = python311Packages.fetchPypi {
            inherit pname version format;
            dist = "py2.py3";
            python = "py2.py3";
            platform = "any";
            sha256 = "8af12cb2d1fc0d1a9509a620e7589daf173714e99f08aaf090a4748ff20b45c8";
          };

          propagatedBuildInputs = with python311Packages; [
            numpy
            protobuf
          ];

          doCheck = false;
        };
        tensorflow-estimator = python311Packages.buildPythonPackage rec {
          pname = "tensorflow_estimator";
          version = "2.13.0";
          format = "wheel";

          src = python311Packages.fetchPypi {
            inherit pname version format;
            dist = "py2.py3";
            python = "py2.py3";
            platform = "any";
            sha256 = "6f868284eaa654ae3aa7cacdbef2175d0909df9fcf11374f5166f8bf475952aa";
          };

          propagatedBuildInputs = with python311Packages; [
            mock
            numpy
            absl-py
          ];

          doCheck = false;
        };
        tensorflow-io = python311Packages.buildPythonPackage rec {
          pname = "tensorflow_io";
          version = "0.34.0";
          format = "wheel";

          src = python311Packages.fetchPypi {
            inherit pname version format;
            dist = "cp311";
            python = "cp311";
            abi = "cp311";
            platform = "manylinux_2_12_x86_64.manylinux2010_x86_64";
            sha256 = "2128203e29bc02b69e73aaddcd3c8300de515f5162163b3562a792bc5826b375";
          };

          propagatedBuildInputs = with python311Packages; [
            # TODO
          ];

          doCheck = false;
        };
      in
        [
          ffmpeg

          python311
          tensorflow-hub
          tensorflow-io

          # TODO: for tensorflow, should be packaged upstream
          tensorflow-estimator
          python311Packages.keras
        ]
        ++ (with python311Packages; [
          numpy
          torch
          torchaudio
          librosa
          more-itertools
          pytube
          pandas
          matplotlib
          tensorflow
          ipython
          soundfile
        ]);
    };
  };
}
