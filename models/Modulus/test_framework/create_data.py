import json

def list2json():
    string = "examples/annular_ring/annular_ring/annular_ring.py examples/annular_ring/annular_ring_equation_instancing/annular_ring.py examples/annular_ring/annular_ring_gradient_enhanced/annular_ring_gradient_enhanced.py examples/annular_ring/annular_ring_hardBC/annular_ring_hardBC.py examples/annular_ring/annular_ring_parameterized/annular_ring_parameterized.py examples/anti_derivative/data_informed.py examples/anti_derivative/physics_informed.py examples/bracket/bracket.py examples/chip_2d/chip_2d_solid_fluid_heat_transfer_flow.py examples/chip_2d/chip_2d_solid_fluid_heat_transfer_heat.py examples/chip_2d/chip_2d_solid_solid_heat_transfer.py examples/chip_2d/chip_2d.py examples/cylinder/cylinder_2d.py examples/darcy/darcy_AFNO.py examples/darcy/darcy_DeepO.py examples/darcy/darcy_FNO_lazy.py examples/darcy/darcy_FNO.py examples/darcy/darcy_PINO.py examples/fuselage_panel/panel.py examples/helmholtz/helmholtz_hardBC.py examples/helmholtz/helmholtz_ntk.py examples/helmholtz/helmholtz.py examples/ldc/ldc_2d_domain_decomposition_fbpinn.py examples/ldc/ldc_2d_domain_decomposition.py examples/ldc/ldc_2d_importance_sampling.py examples/ldc/ldc_2d_zeroEq.py examples/ldc/ldc_2d.py examples/limerock/limerock_hFTB/limerock_flow.py examples/limerock/limerock_hFTB/limerock_thermal.py examples/limerock/limerock_transfer_learning/limerock_flow.py examples/ode_spring_mass/spring_mass_solver.py examples/seismic_wave/wave_2d.py examples/surface_pde/sphere/sphere.py examples/taylor_green/taylor_green_causal.py examples/taylor_green/taylor_green.py examples/three_fin_2d/heat_sink_inverse.py examples/three_fin_2d/heat_sink.py examples/three_fin_3d/three_fin_flow.py examples/three_fin_3d/three_fin_thermal.py examples/turbulent_channel/2d/re590_k_ep_LS.py examples/turbulent_channel/2d/re590_k_om_LS.py examples/turbulent_channel/2d_std_wf/re590_k_ep.py examples/turbulent_channel/2d_std_wf/re590_k_om.py examples/turbulent_channel/2d_std_wf/u_tau_lookup.py examples/wave_equation/wave_1d_causal.py examples/wave_equation/wave_1d.py examples/wave_equation/wave_inverse.py examples/waveguide/cavity_2D/waveguide2D_TMz.py examples/waveguide/cavity_3D/waveguide3D.py examples/waveguide/slab_2D/slab_2D.py examples/waveguide/slab_3D/slab_3D.py"
    array = string.split()
    model_json ={}
    for model_pwd in array:
        array2 = model_pwd.split("/")
        json_key = array2[-1].split(".")[0]
        if json_key not in model_json:
            model_json[json_key] = model_pwd
        else:
            print(f"model {json_key} already exists")
    print(model_json)

def json2json(json_file_path):
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # 修改键
    modified_data = {}
    for old_key in data:
        # 示例：将键名转换为大写
        new_key = data[old_key].replace("/", "-").split(".")[0]
        modified_data[new_key] = data[old_key]
    
    # 将修改后的数据写回原始 JSON 文件
    with open(json_file_path, 'w', encoding='utf-8') as file:
        json.dump(modified_data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # list2json()
    json2json("./model.json")