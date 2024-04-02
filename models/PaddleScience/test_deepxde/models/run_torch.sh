rm -rf test_*.py
export DDE_BACKEND=pytorch
cases=`find ../../deepxde/examples/ -name "*.py" | sort `
ignore="dataset.py func_uncertainty.py func.py mf_dataset.py \
        mf_func.py antiderivative_aligned.py antiderivative_unaligned.py \
        Euler_beam.py Klein_Gordon.py \
        fractional_diffusion_1d.py fractional_Poisson_1d_inverse.py \
        fractional_Poisson_1d.py fractional_Poisson_2d_inverse.py \
        fractional_Poisson_2d.py fractional_Poisson_3d.py ide.py \
        Lorenz_inverse_forced.py Poisson_Dirichlet_1d_exactBC.py \
        Poisson_multiscale_1d.py wave_1d.py \
        "
serial_bug=0
bug=0
echo "============ failed cases =============" > result.txt
for file in ${cases}
do
echo serial ${file} test
if [[ ${ignore} =~ ${file##*/} ]]; then
    echo "skip"
else
    python3.7 ${file}
    if [ $? -ne 0 ]; then
        echo ${file} >> result.txt
        bug=`expr ${bug} + 1`
        serial_bug=`expr ${serial_bug} + 1`
    fi
fi
done
echo "serial bugs: "${serial_bug} >> result.txt