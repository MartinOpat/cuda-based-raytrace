with import <nixpkgs> { config.allowUnfree = true; };

stdenv.mkDerivation {
	name = "cuda-raytracer";
	src = ./.;
	nativeBuildInputs = with cudaPackages; [cmake autoAddDriverRunpath autoPatchelfHook ];
	buildInputs = with pkgs; with cudaPackages; [ libGL glfw netcdf netcdfcxx4 cuda_nvcc cuda_cudart cuda_cccl libcurand];

	postConfigure = ''
		export netCDFCxx_DIR=${netcdfcxx4}/lib/cmake/netCDFCxx
	'';

	installTargets = "preinstall";

	postInstall = ''
	    mkdir -p $out/bin
     	cp cuda-raytracer $out/bin
	'';

    LD_LIBRARY_PATH="/run/opengl-driver/lib/";
}
