This is implementation of physics-informed VAE of the paper:
PI-VAE: Physics-Informed Variational Auto-Encoder for stochastic differential equations( https://arxiv.org/abs/2203.11363 ) \
If you make advantage of the PI-VAE in your research, please consider citing our paper in your manuscript.

# 1. SDE training data construction

## 1.2 low dimension
> cd lib
> python training_data.py --case='ODE' --kl=1.0 --fl=0.2 --mesh_size=400

## 1.2 high dimension imbalance
> cd lib
> python training_data.py --case='ODE_khigh' --kl=0.02 --fl=1.0 --mesh_size=400 --data_size=5000
> python training_data.py --case='ODE_fhigh' --kl=1.0 --fl=0.02 --mesh_size=400 --data_size=5000



# 2. Numerical Experiments

### 2.1 ODE problem
> cd examples
> python SDE.py --case='ODE' --u_sensor=2 --k_sensor=17 --f_sensor=21 --mesh_size=400 --epoch=2000
> python SDE.py --case='ODE' --u_sensor=6 --k_sensor=6 --f_sensor=21 --mesh_size=400 --epoch=2000
> python SDE.py --case='ODE' --u_sensor=11 --k_sensor=1 --f_sensor=21 --mesh_size=400 --epoch=2000

### 2.2 high dimension problem
> cd examples
> python SDE.py --case='ODE_khigh' --u_sensor=2 --k_sensor=51 --f_sensor=21 --mesh_size=400 --epoch=2000 --data_size=5000
> python SDE.py --case='ODE_fhigh' --u_sensor=2 --k_sensor=17 --f_sensor=51 --mesh_size=400 --epoch=2000 --data_size=5000

