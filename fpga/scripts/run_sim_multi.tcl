set origin_dir [file normalize [pwd]]
set script_dir [file normalize [file dirname [info script]]]
cd $script_dir/..

set rtl_files [list \
    rtl/snn_fc_top.sv \
    sim/spike_fc_tb_multi.sv \
]

foreach file $rtl_files {
    puts "Compiling $file"
    exec xvlog -sv $file
}

exec xelab spike_fc_tb_multi -s spike_fc_sim_multi
exec xsim spike_fc_sim_multi -runall

cd $origin_dir

