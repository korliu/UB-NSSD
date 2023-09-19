{
  description = "UB-NSSD development environment";

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShellNoCC {
      LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

      # TODO: https://cyberchris.xyz/posts/nix-python-pyright/
      buildInputs = with pkgs; let
        whisper-at = python310Packages.buildPythonPackage rec {
          pname = "whisper-at";
          version = "0.5";

          src = python310Packages.fetchPypi {
            inherit pname version;
            sha256 = "f685f7bd71ab262b7307d9bd1e0eea5eac75b9ba34add422ea9e97937f21b19f";
          };

          propagatedBuildInputs = with python310Packages; [
            numba
            numpy
            torch-bin
            tqdm
            more-itertools
            tiktoken
            openai-triton-bin
          ];

          doCheck = false;
        };
        torcheval = python310Packages.buildPythonPackage rec {
          pname = "torcheval";
          version = "0.0.6";

          src = fetchFromGitHub {
            owner = "pytorch";
            repo = "${pname}";
            rev = "${version}";
            sha256 = "FnMSPU8tjXegLH4speeyD8UDrKSvjf8STftt7aXTuJI=";
          };

          propagatedBuildInputs = with python310Packages; [
            typing-extensions
          ];

          doCheck = false;
        };
        pytube = python310Packages.buildPythonPackage rec {
          pname = "pytube";
          version = "15.0.0";

          src = fetchFromGitHub {
            owner = "${pname}";
            repo = "${pname}";
            rev = "v${version}";
            sha256 = "Nvs/YlOjk/P5nd1kpUnCM2n6yiEaqZP830UQI0Ug1rk=";
          };

          doCheck = false;
        };
      in [
        ffmpeg

        python310
        python310Packages.matplotlib
        # python310Packages.pytube
        python310Packages.pandas
        python310Packages.torchaudio-bin
        whisper-at
        torcheval
        pytube
      ];
    };
  };
}
