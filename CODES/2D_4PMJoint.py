# 4PM inversion routine ( Wagner et al., 2019)
import sys
import numpy as np

import pybert as pb
import pygimli as pg
pg.verbose = print # temporary
import pygimli.meshtools as mt
import numpy as np
from fpinv import FourPhaseModel, NN_interpolate, JointInv, JointMod
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime.ratools import createGradientModel2D

####################################################################################### INVERSION SETTINGS ##################################################################################################
erte    = 0.05	  # 5%:   relative error on ert data
rste    = 0.005   # 5 ms: absolute error on rst data
zWeight = 1       # equal smoothing in lateral and vertical direction
maxIter = 15      # maximum number of iterations
lam     = 10     # regularization weight
weighting = False # if you want to weight the data in the joint inversion accordingly to error on observations
#############################################################################################################################################################################################################


####################################################################################### PETROPHYSICAL SETTINGS ##############################################################################################
poro     = 0.3  # porosity assumed
fix_poro = False
poro_min = 0.
poro_max = 0.8
fr_min   = 1 - poro_max
fr_max   = 1 - poro_min
phi      = poro
fpm      = FourPhaseModel(phi=poro, va=300., vi=3500., vw=1500, m=1.3, n=2.5, rhow=100, vr=6000) 
#############################################################################################################################################################################################################


############################################################################### IMPORT DATASET #############################################################################################################
ertData = pb.load("JOTEert.dat")
rstData = pg.DataContainer("JOTErst.dat", "s g")

############################################################################# COMBINE SENSORS FOR CREATING MESH #############################################################################################
print("Number of electrodes:", ertData.sensorCount())
print(ertData)
print("Number of shot/receivers:", rstData.sensorCount())
maxrst = pg.max(pg.x(rstData.sensors()))

# filtering ERT observations
ertData.removeInvalid()
ertData.removeUnusedSensors()
ertData.set("err", pg.RVector(ertData.size(), erte))
ertData.save("ert_filtered.data")

rstData.set("err", pg.RVector(rstData.size(), rste))

# # Remove 17 data points with v_a>1500
# Calculate offset
px = pg.x(rstData.sensorPositions())
gx = np.array([px[int(g)] for g in rstData("g")])
sx = np.array([px[int(s)] for s in rstData("s")])
offset = np.absolute(gx - sx)
va = offset / rstData("t")
rstData.markInvalid((va > 1500))
rstData.removeInvalid()
#########################
rstData.save("rst_filtered.data")
rstData = pg.DataContainer("rst_filtered.data", "s g")

def is_close(pos, data, tolerance=0.1):
    for posi in data.sensorPositions():
        dist = pos.dist(posi)
        if dist <= tolerance:
            return True
    return False
combinedSensors = pg.DataContainer()
for pos in ertData.sensorPositions():
    combinedSensors.createSensor(pos)
for pos in rstData.sensorPositions():
    if is_close(pos, ertData):
        print("Not adding", pos)
    else:
        combinedSensors.createSensor(pos)
combinedSensors.sortSensorsX()
x = pg.x(combinedSensors.sensorPositions()).array()
z = pg.z(combinedSensors.sensorPositions()).array()
np.savetxt("sensors.npy", np.column_stack((x, z)))
print("Number of combined positions:", combinedSensors.sensorCount())
print(combinedSensors)

# CREATE MESH
plc = mt.createParaMeshPLC(combinedSensors, paraDX=0.5, boundary=4, paraBoundary=2, paraDepth=150.,paraMaxCellSize=400.)
mesh = mt.createMesh(plc, quality=34.)
# Set vertical boundaries of box to zero to allow lateral smoothing
for bound in mesh.boundaries():
	if bound.marker() == 20:
                bound.setMarker(0)
mesh.save("mesh.bms")
# Extract inner domain where parameters should be estimated. Outer domain is only needed for ERT forward simulation, not for seismic traveltime calculations.
paraDomain = pg.Mesh(2)
paraDomain.createMeshByMarker(mesh, 2)
paraDomain.save("paraDomain.bms")
#############################################################################################################################################################################################################



############################################################################################# 4PM JOINT INVERSION ###########################################################################################
# Setup managers and equip with meshes
ertData = pb.load("ert_filtered.data")
rstData = pg.DataContainer("rst_filtered.data", "s g")
print(ertData)
ertScheme = pg.DataContainerERT(ertData)
ert = ERTManager()
ert.setMesh(mesh)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()
rst = Refraction(rstData, verbose=True)
ttData = rst.dataContainer
rst.setMesh(paraDomain, secNodes=3)

# Setup joint modeling and inverse operators
JM = JointMod(paraDomain, ert, rst, fpm, fix_poro=False, zWeight=zWeight, fix_ice=False)
data = pg.cat(ttData("t"), ertScheme("rhoa"))
if weighting:
    n_rst = ttData.size()
    n_ert = ertScheme.size()
    avg = (n_rst + n_ert) / 2
    weight_rst = avg / n_rst
    weight_ert = avg / n_ert
else:
    weight_rst = 1
    weight_ert = 1
error = pg.cat( rst.relErrorVals(ttData) / weight_rst, ertScheme("err") / weight_ert)

# Set gradient starting model of f_ice, f_water, f_air = phi/3 and constant median value for rhoa
minvel = 300
maxvel = 5000
startmodel = createGradientModel2D(ttData, paraDomain, minvel, maxvel)
np.savetxt("rst_startmodel.dat", 1 / startmodel)
velstart = 1/ startmodel
rhostart = np.ones_like(velstart) * np.median(ertScheme("rhoa"))
fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
frs[frs <= fr_min] = fr_min + 0.01
frs[frs >= fr_max] = fr_max - 0.01
startmodel = np.concatenate((fws, fis, fas, frs))

# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.01] = 0.01
inv = JointInv(JM, data, error, startmodel, lam=lam, frmin=fr_min, frmax=fr_max, maxIter=maxIter)
inv.setModel(startmodel)

# Run inversion
model = inv.run()
pg.boxprint(("Chi squared fit:", inv.getChi2()), sym="+")

# Save results
fwe, fie, fae, fre = JM.fractions(model)
fsum = fwe + fie + fae + fre

print("Min/Max sum:", min(fsum), max(fsum))

rhoest = JM.fpm.rho(fwe, fie, fae, fre)
velest = 1. / JM.fpm.slowness(fwe, fie, fae, fre)

array_mask = np.array(((fae < 0) | (fae > 1 - fre))
                      | ((fie < 0) | (fie > 1 - fre))
                      | ((fwe < 0) | (fwe > 1 - fre))
                      | ((fre < 0) | (fre > 1))
                      | (fsum > 1.01))

# Creating fwd responses vectores for residual estimates
Rresp=ert.fop.response(rhoest)
Tresp=rst.simulate(paraDomain,(1./velest),rstData,verbose=False)
Rresp=np.array(Rresp)
Tresp=np.array(Tresp('t'))
np.savetxt('Resp_ERT.txt',Rresp)
np.savetxt('Resp_RST.txt',Tresp)

np.savez("joint_inversion_n25_m13.npz" , vel=np.array(velest), rho=np.array(rhoest), fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)

print("#" * 80)
ertchi, _ = JM.ERTchi2(model, error)
rstchi, _ = JM.RSTchi2(model, error, ttData("t"))
print("ERT chi^2", ertchi)
print("RST chi^2", rstchi)
print("#" * 80)

fig = JM.showFit(model)
title = "Overall chi^2 = %.2f" % inv.getChi2()
title += "\nERT chi^2 = %.2f" % ertchi
title += "\nRST chi^2 = %.2f" % rstchi
fig.suptitle(title)
fig.savefig("datafit_s.png", dpi=150)


