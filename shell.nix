let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  };

  python = pkgs.python311.withPackages (ps: with ps; [
    torch 
    torchvision
    torchaudio
    numpy
    pandas
    scikit-learn
    matplotlib
    pillow
    tqdm
  ]);
in

pkgs.mkShell {
  buildInputs = [ 
    python 
  ];
}

