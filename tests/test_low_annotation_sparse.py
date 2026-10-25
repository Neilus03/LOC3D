"""Behavioral tests for partial observations, independent of dense training code."""
import copy
import unittest
import numpy as np
import torch
from delimit3d.tasks.mask3d.sparse_supervision import (select_annotations, SparseAnnotations, observed_targets,
                                SparseMask3DCriterion, geometry_voxel_segments)


def annotation(indices=(0,2,3), instances=(1,2,0), semantics=(5,7,1), n=6, device='cpu'):
    return SparseAnnotations(torch.tensor(indices,dtype=torch.long,device=device),
        torch.tensor(instances,dtype=torch.long,device=device),
        torch.tensor(semantics,dtype=torch.long,device=device),n)

def prediction(device='cpu'):
    torch.manual_seed(7)
    p=dict(segment_mask_logits=torch.randn(5,6,device=device,requires_grad=True),
           object_logits=torch.randn(5,2,device=device,requires_grad=True),
           point_to_segment=torch.arange(6,device=device))
    p['aux_outputs']=[dict(segment_mask_logits=torch.randn(5,6,device=device,requires_grad=True),
                           object_logits=torch.randn(5,2,device=device,requires_grad=True))]
    return p


class SparseTests(unittest.TestCase):
    def test_hidden_mutation_deletion_preserves_boundary_matching_loss_gradients(self):
        segments=[10,99,20,30,99,99]
        groups=[dict(objectId=8,segments=[10],label='chair'),dict(objectId=20,segments=[20],label='table'),
                dict(objectId=3,segments=[30],label='wall'),dict(objectId=77,segments=[99],label='bed')]
        changed=copy.deepcopy(groups[:3]);changed[0]['segments'] += [4321]
        allowed=[]
        for groups_variant,segments_variant in [(groups,segments),(changed,[10,1234,20,30,8765,2345])]:
            a=select_annotations([0,2,3],6,segments_variant,groups_variant,{'chair':5,'table':7,'wall':1,'bed':4})
            allowed.append(a)
        for k in allowed[0]:np.testing.assert_array_equal(allowed[0][k],allowed[1][k])
        traces=[]
        for a in allowed:
            p=prediction();c=SparseMask3DCriterion()
            loss=c([p],[SparseAnnotations(*(torch.from_numpy(a[k]) for k in ['vertex_indices','instance_ids','semantic_ids']),6)])
            loss['loss_total'].backward()
            traces.append((loss['loss_total'].detach(),c.all_matches,
                [v.grad for layer in [p,*p['aux_outputs']] for k,v in layer.items() if k in ['segment_mask_logits','object_logits']]))
        self.assertTrue(torch.equal(traces[0][0],traces[1][0]))
        for x,y in zip(traces[0][2],traces[1][2],strict=True):self.assertTrue(torch.equal(x,y))
        for l,r in zip(traces[0][1],traces[1][1],strict=True):
            for (a,b),(x,y) in zip(l,r,strict=True):self.assertTrue(torch.equal(a,x) and torch.equal(b,y))

    def test_indices_bounds_duplicates_and_order(self):
        g=[dict(objectId=9,segments=[10],label='chair')]
        a=select_annotations([2,0],3,[10,88,10],g,{'chair':5})
        self.assertEqual(a['vertex_indices'].tolist(),[2,0])
        duplicate=select_annotations([0,0],3,[10,88,10],g,{'chair':5})
        self.assertEqual(duplicate['vertex_indices'].tolist(),[0,0])
        for bad in [[-1],[3],[1.0]]:
            with self.assertRaises(ValueError):select_annotations(bad,3,[10,88,10],g,{'chair':5})

    def test_unknown_points_and_unmatched_queries_have_zero_gradient(self):
        p=prediction();c=SparseMask3DCriterion();a=annotation()
        c([p],[a])['loss_total'].backward()
        for layer,matches in zip([p,*p['aux_outputs']],c.all_matches,strict=True):
            self.assertEqual(float(layer['segment_mask_logits'].grad[:,[1,4,5]].abs().sum()),0.)
            unmatched=list(set(range(5))-set(matches[0][0].tolist()))
            self.assertEqual(float(layer['object_logits'].grad[unmatched].abs().sum()),0.)
            self.assertEqual(float(layer['segment_mask_logits'].grad[unmatched].abs().sum()),0.)

    def test_unobserved_object_is_not_a_target(self):
        t=observed_targets(annotation(),torch.arange(6))
        self.assertEqual(t.instance_ids.tolist(),[1,2])
        self.assertEqual(t.columns.tolist(),[0,2,3])
        self.assertIsNone(t.statistics['unknown_instances'])

    def test_duplicate_instance_annotations_and_voxel_conflicts(self):
        a=annotation(indices=(0,1,2,3,4),instances=(1,1,2,3,4),semantics=(5,5,7,4,5))
        t=observed_targets(a,torch.tensor([0,0,1,1,2,3]))
        self.assertEqual(t.instance_ids.tolist(),[1,4])
        self.assertEqual(t.statistics['retained_annotations'],3)
        self.assertEqual(t.statistics['coalesced_annotations'],1)
        self.assertEqual(t.statistics['conflicting_annotations'],2)
        self.assertEqual(t.masks.tolist(),[[1.,0.],[0.,1.]])

    def test_same_identity_different_semantic_conflicts(self):
        a=annotation(indices=(0,1),instances=(1,1),semantics=(5,7))
        t=observed_targets(a,torch.tensor([0,0,1,2,3,4]))
        self.assertEqual(len(t.columns),0)

    def test_empty_and_only_background_are_zero_not_no_object(self):
        for a in [annotation(indices=(),instances=(),semantics=()),
                  annotation(indices=(1,),instances=(0,),semantics=(1,)),
                  annotation(indices=(1,),instances=(-1,),semantics=(-1,))]:
            p=prediction();loss=SparseMask3DCriterion()([p],[a])['loss_total']
            self.assertEqual(float(loss.detach()),0.);loss.backward()
            for layer in [p,*p['aux_outputs']]:
                for key in ['segment_mask_logits','object_logits']:
                    self.assertTrue(torch.isfinite(layer[key].grad).all())
                    self.assertEqual(float(layer[key].grad.abs().sum()),0.)

    def test_annotations_survive_nonrepresentative_voxels_and_batch_offsets(self):
        xyz=torch.tensor([[0.,0.,0.],[.01,0.,0.],[.11,0.,0.],[0.,0.,0.],[.12,0.,0.]])
        segments=geometry_voxel_segments(xyz,torch.tensor([3,5]))
        self.assertEqual(segments.tolist(),[0,0,1,0,1])
        a=annotation(indices=(1,),instances=(1,),semantics=(5,),n=3)
        t=observed_targets(a,segments[:3]);self.assertEqual(t.columns.tolist(),[0])
        self.assertEqual(t.statistics['retained_annotations'],1)

    def test_more_observed_objects_than_queries_fails_closed(self):
        a=annotation(indices=tuple(range(6)),instances=tuple(range(1,7)),semantics=(5,)*6)
        with self.assertRaises(ValueError):SparseMask3DCriterion()([prediction()],[a])

    @unittest.skipUnless(torch.cuda.is_available(),'requires real GPU allocation')
    def test_gpu_sparse_matching_loss_and_gradient_invariance(self):
        a=annotation(device='cuda');p=prediction('cuda');c=SparseMask3DCriterion()
        loss=c([p],[a])['loss_total'];loss.backward()
        grad=p['segment_mask_logits'].grad.clone()
        p2=prediction('cuda');loss2=c([p2],[a])['loss_total'];loss2.backward()
        self.assertTrue(torch.equal(loss,loss2));self.assertTrue(torch.equal(grad,p2['segment_mask_logits'].grad))
        self.assertEqual(float(grad[:,[1,4,5]].abs().sum()),0.)

if __name__=='__main__':unittest.main(verbosity=2)
