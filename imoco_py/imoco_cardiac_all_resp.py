import argparse 
import os
import scipy.io as sio
import sigpy as sp
import scipy.ndimage as ndimage_c
import numpy as np

import sys
sys.path.append("./sigpy_e/")
import sigpy_e.cfl as cfl

import sigpy_e.ext as ext
import sigpy_e.prox as prox
import sigpy_e.reg as reg
from sigpy_e.linop_e import NFTs,Diags,DLD,Vstacks
import sigpy.mri as mr

if __name__ == '__main__':

    ## IO parameters
    parser = argparse.ArgumentParser(description='iterative motion compensation recon for FB spiral cardiac data.')

    parser.add_argument('--n_ref', type=int, default=-1,
                        help='reference frame, default is -1.')
    parser.add_argument('--reg_flag', type=int, default=0,
                        help='derive motion field from registration')
    parser.add_argument('--lambda_TV', type=float, default=5e-2,
                        help='low rank regularization, default is 0.05')
    parser.add_argument('--iner_iter', type=int, default=15,
                        help='Num of inner iterations.')
    parser.add_argument('--outer_iter', type=int, default=20,
                        help='Num of outer iterations.')
    parser.add_argument('--device', type=int, default=0,
                        help='Computing device.')
    parser.add_argument('--dir', type=str,
                        help='Working directory for data/traj/results.')
    args = parser.parse_args()

    #
    fname = args.dir
    lambda_TV = args.lambda_TV
    device = args.device
    outer_iter = args.outer_iter
    iner_iter = args.iner_iter
    n_ref = args.n_ref
    reg_flag = args.reg_flag

    ## data loading
    data_all = cfl.read_cfl(os.path.join(fname, 'ksp'))
    traj_all = np.real(cfl.read_cfl(os.path.join(fname, 'traj')))
    kdens = np.squeeze(sio.loadmat(os.path.join(fname, 'kdens.mat'))['kdens'])
    # pre-calculated sens map
    mps = cfl.read_cfl(os.path.join(fname, 'sens'))
    # pre-calculated XD-grasp recon
    imgLs = cfl.read_cfl(os.path.join(fname, 'recon_l1_1_tv_5')) # [echo, nresp, ncard, nz, ny, nx]
    nresp, ncard, ne, nCoil, npe, nfe = np.squeeze(data_all).shape

    ## flatten resp and card phases
    nphase = ncard*nresp
    traj = np.squeeze(traj_all)[:,:,np.newaxis,np.newaxis,np.newaxis,...] # [resp, card, 1, 1, 1, npe, nfe, 1]
    data = np.squeeze(data_all)[:,:,np.newaxis,...,np.newaxis] # [resp, card, 1, echo, coil, npe, nfe, 1]
    # need to take sqrt of the spiral output kdens for dcf
    dcf = np.tile(np.sqrt(kdens), (nresp, ncard, npe, 1))[:,:,np.newaxis,np.newaxis,np.newaxis,...,np.newaxis] # [nresp, ncard, 1, 1, 1, npe, nfe, 1]
    # reshape the input to stack ncard along nresp, as [resp*card, 1, echo, coil, npe, nfe, 1]
    data  = data.reshape((-1,)+data.shape[2:])
    traj  = traj.reshape((-1,)+traj.shape[2:])
    dcf  = dcf.reshape((-1,)+dcf.shape[2:])
    traj[...,[0,2]] = traj[...,[2,0]] # row-major data struct so the order is (kz, ky, kx)
    imgLs = np.transpose(imgLs, (1,2,0,3,4,5)) # [nresp, ncard, echo, nz, ny, nx]
    imgL = imgLs.reshape((-1,)+imgLs.shape[2:]) # [nresp*ncard, echo, nz, ny, nx]
    tshape = imgL.shape[2::] # output size in image domain

    ## imoco recon params
    lambda_TV = 0.05
    outer_iter = 20

    ## calibration
    print('Calibration...')
    S = sp.linop.Multiply(tshape, mps)

    ## registration
    print('Registration...')
    M_fields = []
    iM_fields = []
    imgL_comb_mag = np.sqrt(np.sum(np.abs(imgL)**2,1))
    # nref = ncard - 1 # take the last cardiac phase of the first resp phase
    if reg_flag == 1:
        for i in range(nphase):
            M_field, iM_field = reg.ANTsReg(imgL_comb_mag[n_ref], imgL_comb_mag[i])
            M_fields.append(M_field)
            iM_fields.append(iM_field)
        M_fields = np.asarray(M_fields)
        iM_fields = np.asarray(iM_fields)
        np.save(os.path.join(fname, 'M_mr.npy'),M_fields)
        np.save(os.path.join(fname, 'iM_mr.npy'),iM_fields)
    else:
        M_fields = np.load(os.path.join(fname,'M_mr.npy'))
        iM_fields = np.load(os.path.join(fname,'iM_mr.npy'))

    # numpy array to list
    iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
    M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

    ######## TODO scale M_field
    print('Motion Field scaling...')
    M_fields = [reg.M_scale(M,tshape) for M in M_fields]
    iM_fields = [reg.M_scale(M,tshape) for M in iM_fields]

    ## low rank
    print('Prep...')
    Ms = []
    M0s = []
    for i in range(nphase):
        # M = reg.interp_op(tshape,iM_fields[i],M_fields[i])
        M = reg.interp_op(tshape,M_fields[i])
        M0 = reg.interp_op(tshape,np.zeros(tshape+(3,)))
        M = DLD(M,device=sp.Device(device))
        M0 = DLD(M0,device=sp.Device(device))
        Ms.append(M)
        M0s.append(M0)
    Ms = Diags(Ms,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)
    M0s = Diags(M0s,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)

    PFTSMs = []
    Is = []
    for i in range(nphase):
        Is.append(sp.linop.Identity(tshape))
        FTs = NFTs((nCoil,)+tshape,traj[i,0,0,0,...],device=sp.Device(device))
        M = reg.interp_op(tshape,M_fields[i])
        M = DLD(M,device=sp.Device(device))
        W = sp.linop.Multiply((nCoil,npe,nfe,),dcf[i,0,0,0,:,:,0]) 
        FTSM = W*FTs*S*M
        PFTSMs.append(FTSM)
    PFTSMs = Diags(PFTSMs,oshape=(nphase,nCoil,npe,nfe,),ishape=(nphase,)+tshape)*Vstacks(Is,ishape=tshape,oshape=(nphase,)+tshape)
    
    ## precondition
    print('Preconditioner calculation...')
    tmp = PFTSMs.H*PFTSMs*np.complex64(np.ones(tshape))
    L=np.mean(np.abs(tmp))
    wdata = data[:,0,...,0]*dcf[:,0,...,0]*1e4
    
    TV = sp.linop.FiniteDifference(PFTSMs.ishape,axes = (0,1,2))
    ####### debug
    print('TV dim:{}'.format(TV.oshape))
    proxg = sp.prox.UnitaryTransform(sp.prox.L1Reg(TV.oshape, lambda_TV), TV)
    
    # ADMM
    print('Recon...')
    sigma = 0.4
    tau = 0.4
    Xs = []
    for ie in range(ne):
        alpha = np.max(np.abs(PFTSMs.H*wdata[:,ie,...]))
        ###### debug
        print('alpha:{}'.format(alpha))
        X = np.zeros(tshape,dtype=np.complex64)
        p = np.zeros_like(wdata[:,ie,...])
        X0 = np.zeros_like(X)
        q = np.zeros((3,)+tshape,dtype=np.complex64)
        for i in range(outer_iter):
            p = (p + sigma*(PFTSMs*X-wdata[:,ie,...]))/(1+sigma)
            q = (q + sigma*TV*X)
            q = q/(np.maximum(np.abs(q),alpha)/alpha)
            X0 = X
            X = X-tau*(1/L*PFTSMs.H*p + lambda_TV*TV.H*q)
            print('outer iter:{}, res:{}'.format(i,np.linalg.norm(X-X0)/np.linalg.norm(X)))
        Xs.append(X)
        
    # np.save(os.path.join(fname, 'imoco_allphase.npy'), Xs)
    Xs = np.stack(Xs, 0)
    cfl.write_cfl(os.path.join(fname, 'recon_imoco_all'), Xs)