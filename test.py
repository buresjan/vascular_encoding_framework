from vef_scripts.vef_scripts.case_io import load_vascular_encoding
enc = load_vascular_encoding("playground/conduit_case")
vol = enc.make_volume_mesh(tau_res=60, theta_res=60, rho_res=6, scheme="cylindrical")
vol.save("playground/conduit_case/Encoding/encoding_volume.vtm", binary=True)
