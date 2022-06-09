Search.setIndex({docnames:["index","modules/attack","modules/dataset","modules/defense","modules/functional","modules/nn","modules/root","modules/training","modules/utils","notes/installation"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.viewcode":1,sphinx:56},filenames:["index.rst","modules/attack.rst","modules/dataset.rst","modules/defense.rst","modules/functional.rst","modules/nn.rst","modules/root.rst","modules/training.rst","modules/utils.rst","notes/installation.rst"],objects:{"graphwar.attack":[[1,1,1,"","Attacker"],[1,1,1,"","FlipAttacker"],[1,0,0,"-","backdoor"],[1,0,0,"-","injection"],[1,0,0,"-","targeted"],[1,0,0,"-","untargeted"]],"graphwar.attack.Attacker":[[1,2,1,"","attack"],[1,2,1,"","data"],[1,3,1,"","edge_index"],[1,3,1,"","edge_weight"],[1,3,1,"","feat"],[1,3,1,"","label"],[1,3,1,"","max_perturbations"],[1,2,1,"","reset"],[1,2,1,"","set_max_perturbations"]],"graphwar.attack.FlipAttacker":[[1,2,1,"","add_edge"],[1,2,1,"","add_feat"],[1,2,1,"","added_edges"],[1,2,1,"","added_feats"],[1,2,1,"","data"],[1,2,1,"","edge_flips"],[1,2,1,"","feat_flips"],[1,2,1,"","is_legal_edge"],[1,2,1,"","is_singleton_edge"],[1,2,1,"","remove_edge"],[1,2,1,"","remove_feat"],[1,2,1,"","removed_edges"],[1,2,1,"","removed_feats"],[1,2,1,"","reset"],[1,2,1,"","set_allow_feature_attack"],[1,2,1,"","set_allow_singleton"],[1,2,1,"","set_allow_structure_attack"]],"graphwar.attack.backdoor":[[1,1,1,"","BackdoorAttacker"],[1,1,1,"","FGBackdoor"],[1,1,1,"","LGCBackdoor"]],"graphwar.attack.backdoor.BackdoorAttacker":[[1,2,1,"","attack"],[1,2,1,"","data"],[1,2,1,"","reset"],[1,2,1,"","trigger"]],"graphwar.attack.backdoor.FGBackdoor":[[1,2,1,"","attack"],[1,2,1,"","setup_surrogate"]],"graphwar.attack.backdoor.LGCBackdoor":[[1,2,1,"","attack"],[1,2,1,"","get_feat_perturbations"],[1,2,1,"","setup_surrogate"]],"graphwar.attack.injection":[[1,1,1,"","AdvInjection"],[1,1,1,"","InjectionAttacker"],[1,1,1,"","RandomInjection"]],"graphwar.attack.injection.AdvInjection":[[1,2,1,"","attack"],[1,2,1,"","compute_gradients"]],"graphwar.attack.injection.InjectionAttacker":[[1,2,1,"","added_edges"],[1,2,1,"","added_feats"],[1,2,1,"","added_nodes"],[1,2,1,"","attack"],[1,2,1,"","data"],[1,2,1,"","edge_flips"],[1,2,1,"","inject_edge"],[1,2,1,"","inject_edges"],[1,2,1,"","inject_feat"],[1,2,1,"","inject_node"],[1,2,1,"","injected_edges"],[1,2,1,"","injected_feats"],[1,2,1,"","injected_nodes"],[1,2,1,"","reset"]],"graphwar.attack.injection.RandomInjection":[[1,2,1,"","attack"]],"graphwar.attack.targeted":[[1,1,1,"","DICEAttack"],[1,1,1,"","FGAttack"],[1,1,1,"","GFAttack"],[1,1,1,"","IGAttack"],[1,1,1,"","Nettack"],[1,1,1,"","RandomAttack"],[1,1,1,"","SGAttack"],[1,1,1,"","TargetedAttacker"]],"graphwar.attack.targeted.DICEAttack":[[1,2,1,"","get_added_edge"],[1,2,1,"","get_removed_edge"]],"graphwar.attack.targeted.FGAttack":[[1,2,1,"","attack"],[1,2,1,"","compute_gradients"],[1,2,1,"","feature_score"],[1,2,1,"","reset"],[1,2,1,"","structure_score"]],"graphwar.attack.targeted.GFAttack":[[1,2,1,"","attack"],[1,2,1,"","get_candidate_edges"],[1,2,1,"","structure_score"]],"graphwar.attack.targeted.IGAttack":[[1,2,1,"","attack"],[1,2,1,"","compute_feature_gradients"],[1,2,1,"","compute_structure_gradients"],[1,2,1,"","get_candidate_edges"],[1,2,1,"","get_candidate_features"],[1,2,1,"","get_feature_importance"],[1,2,1,"","get_link_importance"]],"graphwar.attack.targeted.Nettack":[[1,2,1,"","attack"],[1,2,1,"","compute_XW"],[1,2,1,"","compute_cooccurrence_constraint"],[1,2,1,"","compute_logits"],[1,2,1,"","compute_new_a_hat_uv"],[1,2,1,"","feature_scores"],[1,2,1,"","get_attacker_nodes"],[1,2,1,"","get_candidate_edges"],[1,2,1,"","gradient_wrt_x"],[1,2,1,"","reset"],[1,2,1,"","setup_surrogate"],[1,2,1,"","strongest_wrong_class"],[1,2,1,"","structure_score"]],"graphwar.attack.targeted.RandomAttack":[[1,2,1,"","attack"],[1,2,1,"","get_added_edge"],[1,2,1,"","get_removed_edge"]],"graphwar.attack.targeted.SGAttack":[[1,2,1,"","attack"],[1,2,1,"","compute_gradients"],[1,2,1,"","get_subgraph"],[1,2,1,"","get_top_attackers"],[1,2,1,"","set_normalize"],[1,2,1,"","setup_surrogate"],[1,2,1,"","strongest_wrong_class"],[1,2,1,"","subgraph_processing"]],"graphwar.attack.targeted.TargetedAttacker":[[1,2,1,"","attack"],[1,2,1,"","is_legal_edge"],[1,2,1,"","reset"]],"graphwar.attack.untargeted":[[1,1,1,"","DICEAttack"],[1,1,1,"","FGAttack"],[1,1,1,"","IGAttack"],[1,1,1,"","Metattack"],[1,1,1,"","MinmaxAttack"],[1,1,1,"","PGDAttack"],[1,1,1,"","RandomAttack"],[1,1,1,"","UntargetedAttacker"]],"graphwar.attack.untargeted.DICEAttack":[[1,2,1,"","get_added_edge"],[1,2,1,"","get_removed_edge"]],"graphwar.attack.untargeted.FGAttack":[[1,2,1,"","attack"],[1,2,1,"","compute_gradients"],[1,2,1,"","feature_score"],[1,2,1,"","reset"],[1,2,1,"","setup_surrogate"],[1,2,1,"","structure_score"]],"graphwar.attack.untargeted.IGAttack":[[1,2,1,"","attack"],[1,2,1,"","compute_feature_gradients"],[1,2,1,"","compute_structure_gradients"],[1,2,1,"","feature_score"],[1,2,1,"","get_feature_importance"],[1,2,1,"","get_link_importance"],[1,2,1,"","setup_surrogate"],[1,2,1,"","structure_score"]],"graphwar.attack.untargeted.Metattack":[[1,2,1,"","attack"],[1,2,1,"","clip"],[1,2,1,"","compute_gradients"],[1,2,1,"","feature_score"],[1,2,1,"","forward"],[1,2,1,"","get_perturbed_adj"],[1,2,1,"","get_perturbed_feat"],[1,2,1,"","inner_train"],[1,2,1,"","reset"],[1,2,1,"","reset_parameters"],[1,2,1,"","setup_surrogate"],[1,2,1,"","structure_score"]],"graphwar.attack.untargeted.MinmaxAttack":[[1,2,1,"","attack"],[1,2,1,"","reset"],[1,2,1,"","setup_surrogate"]],"graphwar.attack.untargeted.PGDAttack":[[1,2,1,"","attack"],[1,2,1,"","bernoulli_sample"],[1,2,1,"","bisection"],[1,2,1,"","clip"],[1,2,1,"","compute_gradients"],[1,2,1,"","compute_loss"],[1,2,1,"","config_C"],[1,2,1,"","get_perturbed_adj"],[1,2,1,"","projection"],[1,2,1,"","reset"],[1,2,1,"","setup_surrogate"]],"graphwar.attack.untargeted.RandomAttack":[[1,2,1,"","attack"],[1,2,1,"","get_added_edge"],[1,2,1,"","get_removed_edge"]],"graphwar.attack.untargeted.UntargetedAttacker":[[1,2,1,"","attack"],[1,2,1,"","reset"]],"graphwar.dataset":[[2,1,1,"","GraphWarDataset"]],"graphwar.dataset.GraphWarDataset":[[2,2,1,"","available_datasets"]],"graphwar.defense":[[3,1,1,"","CosinePurification"],[3,1,1,"","DegreeGUARD"],[3,1,1,"","EigenDecomposition"],[3,1,1,"","GNNGUARD"],[3,1,1,"","GUARD"],[3,1,1,"","JaccardPurification"],[3,1,1,"","RandomGUARD"],[3,1,1,"","SVDPurification"],[3,1,1,"","UniversalDefense"]],"graphwar.defense.DegreeGUARD":[[3,4,1,"","training"]],"graphwar.defense.GNNGUARD":[[3,2,1,"","extra_repr"],[3,2,1,"","forward"],[3,4,1,"","training"]],"graphwar.defense.GUARD":[[3,2,1,"","setup_surrogate"],[3,4,1,"","training"]],"graphwar.defense.RandomGUARD":[[3,4,1,"","training"]],"graphwar.defense.UniversalDefense":[[3,2,1,"","anchors"],[3,2,1,"","forward"],[3,2,1,"","patch"],[3,2,1,"","removed_edges"],[3,4,1,"","training"]],"graphwar.functional":[[4,5,1,"","drop_edge"],[4,5,1,"","drop_node"],[4,5,1,"","drop_path"],[4,5,1,"","spmm"],[4,5,1,"","to_dense_adj"],[4,5,1,"","to_sparse_tensor"]],"graphwar.nn":[[5,0,0,"-","layers"],[5,0,0,"-","models"]],"graphwar.nn.layers":[[5,1,1,"","AdaptiveConv"],[5,1,1,"","DAGNNConv"],[5,1,1,"","DropEdge"],[5,1,1,"","DropNode"],[5,1,1,"","DropPath"],[5,1,1,"","ElasticConv"],[5,1,1,"","GCNConv"],[5,1,1,"","MedianConv"],[5,1,1,"","RobustConv"],[5,1,1,"","SATConv"],[5,1,1,"","SGConv"],[5,1,1,"","SSGConv"],[5,1,1,"","Sequential"],[5,1,1,"","SoftMedianConv"],[5,1,1,"","TAGConv"]],"graphwar.nn.layers.AdaptiveConv":[[5,2,1,"","amp_forward"],[5,2,1,"","compute_LX"],[5,2,1,"","forward"],[5,2,1,"","proximal_L21"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.layers.DAGNNConv":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.layers.DropEdge":[[5,2,1,"","forward"],[5,4,1,"","training"]],"graphwar.nn.layers.DropNode":[[5,2,1,"","forward"],[5,4,1,"","training"]],"graphwar.nn.layers.DropPath":[[5,2,1,"","forward"],[5,4,1,"","training"]],"graphwar.nn.layers.ElasticConv":[[5,2,1,"","L1_projection"],[5,2,1,"","L21_projection"],[5,2,1,"","cache_clear"],[5,2,1,"","emp_forward"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"]],"graphwar.nn.layers.GCNConv":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.layers.MedianConv":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.layers.RobustConv":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.layers.SATConv":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.layers.SGConv":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"]],"graphwar.nn.layers.SSGConv":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"]],"graphwar.nn.layers.Sequential":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"]],"graphwar.nn.layers.SoftMedianConv":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"]],"graphwar.nn.layers.TAGConv":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models":[[5,1,1,"","APPNP"],[5,1,1,"","AirGNN"],[5,1,1,"","DAGNN"],[5,1,1,"","ElasticGNN"],[5,1,1,"","GAT"],[5,1,1,"","GCN"],[5,1,1,"","GNNGUARD"],[5,1,1,"","JKNet"],[5,1,1,"","MedianGCN"],[5,1,1,"","RobustGCN"],[5,1,1,"","SAT"],[5,1,1,"","SGC"],[5,1,1,"","SSGC"],[5,1,1,"","SimPGCN"],[5,1,1,"","SoftMedianGCN"],[5,1,1,"","TAGCN"]],"graphwar.nn.models.APPNP":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.AirGNN":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.DAGNN":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.ElasticGNN":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.GAT":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.GCN":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.GNNGUARD":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.JKNet":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.MedianGCN":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.RobustGCN":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.SAT":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.SGC":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.SSGC":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.SimPGCN":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","regression_loss"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.SoftMedianGCN":[[5,2,1,"","cache_clear"],[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.nn.models.TAGCN":[[5,2,1,"","forward"],[5,2,1,"","reset_parameters"],[5,4,1,"","training"]],"graphwar.surrogate":[[6,1,1,"","Surrogate"]],"graphwar.surrogate.Surrogate":[[6,2,1,"","defrozen_surrogate"],[6,2,1,"","estimate_self_training_labels"],[6,2,1,"","freeze_surrogate"],[6,2,1,"","setup_surrogate"]],"graphwar.training":[[7,1,1,"","RobustGCNTrainer"],[7,1,1,"","SATTrainer"],[7,1,1,"","SimPGCNTrainer"],[7,1,1,"","Trainer"],[7,5,1,"","get_trainer"]],"graphwar.training.RobustGCNTrainer":[[7,2,1,"","train_step"]],"graphwar.training.SATTrainer":[[7,2,1,"","train_step"]],"graphwar.training.SimPGCNTrainer":[[7,2,1,"","train_step"]],"graphwar.training.Trainer":[[7,2,1,"","cache_clear"],[7,2,1,"","config_callbacks"],[7,2,1,"","config_optimizer"],[7,2,1,"","config_scheduler"],[7,2,1,"","evaluate"],[7,2,1,"","extra_repr"],[7,2,1,"","fit"],[7,3,1,"","model"],[7,2,1,"","predict"],[7,2,1,"","predict_step"],[7,2,1,"","test_step"],[7,2,1,"","train_step"]],"graphwar.utils":[[8,1,1,"","BunchDict"],[8,1,1,"","CKA"],[8,1,1,"","LikelihoodFilter"],[8,1,1,"","Progbar"],[8,1,1,"","SingletonFilter"],[8,5,1,"","add_edges"],[6,0,0,"-","check"],[8,5,1,"","ego_graph"],[8,5,1,"","flip_edges"],[8,5,1,"","get_logger"],[8,5,1,"","normalize"],[8,5,1,"","overlap"],[8,5,1,"","remove_edges"],[8,5,1,"","repeat"],[8,5,1,"","scipy_normalize"],[6,0,0,"-","seed"],[8,5,1,"","setup_logger"],[8,5,1,"","singleton_filter"],[8,5,1,"","singleton_mask"],[8,5,1,"","split_nodes"],[8,5,1,"","split_nodes_by_classes"],[8,5,1,"","topk"],[8,5,1,"","wrapper"]],"graphwar.utils.BunchDict":[[8,2,1,"","to_tensor"]],"graphwar.utils.CKA":[[8,2,1,"","compare"],[8,2,1,"","export"],[8,2,1,"","plot_results"]],"graphwar.utils.LikelihoodFilter":[[8,2,1,"","compute_alpha"],[8,2,1,"","compute_log_likelihood"],[8,2,1,"","filter_chisquare"],[8,2,1,"","update"],[8,2,1,"","update_Sx"]],"graphwar.utils.Progbar":[[8,2,1,"","add"],[8,2,1,"","format_num"],[8,2,1,"","update"]],"graphwar.utils.SingletonFilter":[[8,2,1,"","update"]],"graphwar.utils.check":[[6,5,1,"","is_edge_index"]],"graphwar.utils.seed":[[6,5,1,"","set_seed"]],graphwar:[[1,0,0,"-","attack"],[2,0,0,"-","dataset"],[3,0,0,"-","defense"],[4,0,0,"-","functional"],[6,0,0,"-","surrogate"],[7,0,0,"-","training"],[8,0,0,"-","utils"]]},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","property","Python property"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:property","4":"py:attribute","5":"py:function"},terms:{"0":[1,2,3,4,5,6,8],"001":1,"004":[1,8],"01":[1,3,5,8],"02":8,"0252":5,"05":[1,8],"0510":5,"0670":5,"1":[1,3,4,5,6,7,8],"10":[1,5],"100":[1,5,7,8],"10138":7,"10903":5,"12":[1,8],"128":1,"1433":7,"15":1,"15327":8,"16":[5,8],"17":[5,8],"1710":5,"18":[1,5,8],"19":[1,3,5,8],"2":[1,3,4,5,6,7,8],"20":[1,3,4,5,8],"200":[1,8],"2010":8,"2020":1,"21":[1,5,8],"2107":5,"22":[1,4,5,8],"22m":8,"2430":5,"2485":7,"2689":5,"2753":5,"2760":5,"2d":4,"2nd":1,"3":[1,4,5,6,7,8],"30":8,"3197":5,"32":[5,8],"3876":5,"3g":8,"4":[4,5,6,7,8],"42":6,"43":8,"4909":5,"5":[1,4,5,8],"50":[1,3,5],"512":3,"5521":5,"6":[5,8],"6029":5,"6338":5,"6385":5,"64":[5,8],"643u":8,"6713":5,"6738":5,"7":8,"8":[1,5,8],"9":1,"9032":5,"9112":5,"9628":5,"\u03bb1":5,"abstract":1,"boolean":[3,8],"case":[1,5,7],"class":[0,2,3,5,6,7,8],"default":[1,2,3,4,5,6,7,8],"do":4,"export":8,"final":8,"float":[1,3,4,5,6,8],"function":[0,1,2,3,5,7,8],"import":[1,2,4,5,6,7,8],"int":[1,3,4,5,6,7,8],"long":8,"new":[8,9],"return":[1,2,3,4,5,6,7,8],"static":[1,2,8],"true":[1,3,4,5,6,8],"while":[1,3,5],A:[1,2,4,5,7,8],As:3,By:8,For:[1,2,5],If:[1,8,9],In:[1,5],It:[1,5,7],NOT:5,One:[5,7],Or:9,That:1,The:[1,2,5,8],There:1,These:2,To:[3,5],__class__:1,__name__:1,_type_:8,a_hat_uv:1,aaai:1,ab:[5,8],abbrev_nam:8,abbrevi:8,acc:7,accept:[3,5,7],access:2,accord:8,act:[3,5,8],activ:[1,3,5,6,9],ad:[1,8],adapt:[1,5,8],adaptiveconv:5,add:[1,3,5,8],add_additional_nod:1,add_edg:[1,8],add_feat:1,add_self_loop:[3,5,8],addbackward0:5,added_edg:1,added_feat:1,added_nod:1,addit:[1,5,7,8],adj:1,adj_chang:1,adj_grad:1,adj_matrix:8,adj_step:1,adj_t:3,adjac:[1,3,4,5,8],adv_inject:1,adversari:[1,2,3,5,8],advinject:1,after:[1,3,5],afterward:[1,3,5],against:[3,5],aggreg:5,airgnn:5,alia:1,align:8,all:[1,2,3,5,6,8,9],allow:[1,3],allow_singleton:3,along:8,alpha:[3,5,8],although:[1,3,5],amp_forward:5,an:[1,2,3,6,8,9],anchor:3,ani:[5,6,8,9],anoth:8,antixk:8,appli:[1,7],appnp:5,approxim:[5,8],ar:[1,2,3,5,6,8],architectur:5,arctan:8,arg:[5,8],argument:[1,5,7,8],arrai:8,arxiv:[1,4,5,8],assum:8,attack:[0,2,3,5,6,8],attack_argu:1,attacker_nod:1,attent:[3,5],attr:[1,7],attribut:8,augment:[4,5],author:[1,5],autoencod:[4,5],avail:2,available_dataset:2,avoid:8,axi:8,b:[1,8],backdoor:0,backdoor_attack:1,backdoorattack:1,ball:5,bar:8,base:[0,2,3,6,8],batch:3,batch_siz:3,batchnorm1d:5,befor:[1,2],being:[1,2],belong:1,below:5,bernoulli_sampl:1,best_wrong_label:1,between:[1,8],bia:[3,5],binar:3,binari:1,binaryz:3,bisect:1,black:1,blob:8,bn:5,bool:[1,3,4,5,6,8],both:[1,3],box:1,budget:1,bunchdict:[1,7,8],c:[1,8],cach:[5,7],cache_clear:[5,7],calcul:1,call:[1,3,5,8],callabl:[2,7,8],callback:[3,7],callbacklist:7,can:[1,5,7,8],candid:[1,8],candidate_edg:1,cannot:1,care:[1,3,5],cat:[5,8],cb:7,cd:9,center:8,cfg:7,challeng:1,check:[1,6],choic:3,choos:1,chosen:[3,4,5],cites:[1,2],cka:8,ckp:3,class_itself:8,classif:[4,5,8],clear:[5,7],clip:1,clone:9,code:[1,5,8],coeffici:5,color:8,com:[2,5,8,9],commun:1,compar:8,compat:[3,5],compon:[2,5],compt:8,comput:[1,3,5,6,8],compute_alpha:8,compute_cooccurrence_constraint:1,compute_feature_gradi:1,compute_gradi:1,compute_log_likelihood:8,compute_logit:1,compute_loss:1,compute_lx:5,compute_new_a_hat_uv:1,compute_structure_gradi:1,compute_xw:1,conduct:1,config_c:1,config_callback:7,config_optim:7,config_schedul:7,connect:[1,2,3],consid:[1,2,8],consist:2,contain:[8,9],continu:[1,3],contrast:[4,5],control:8,conv1:5,conv2:5,conveni:[5,7],convert:[4,8],convolut:[1,4,5,8],cora:[1,2,7],correl:1,correspond:[5,8],cosin:3,cosinepurif:3,count:8,cpu:[1,3,6,7,8],csr:[1,8],csr_matrix:[1,8],cuda:[1,7],cup:1,current:8,custom:[3,4,5,7,8],cutoff:8,cw_loss:1,d:[5,8],d_min:8,d_new:8,d_old:8,dagnn:5,dagnnconv:5,data1:8,data2:8,data:[1,2,3,7,8],dataload:7,dataset:[0,1,3,8],deal:3,debug:8,decomposit:3,decor:8,deep:[1,3,4,5],deeper:5,def:8,defend:[3,5,6],defens:[0,1,5],defin:[1,3,5],defrozen:6,defrozen_surrog:6,degre:[1,3,4,5,8],degreeguard:3,delet:8,denot:[1,3,4,8],dens:[4,5],deriv:[1,8],descend:3,describ:1,design:7,destin:1,detail:2,determin:[1,3],deviat:8,devic:[1,3,6,7,8],dice:1,diceattack:1,dict:[1,7,8],dictionari:8,diego999:5,differ:5,dim:[7,8],dimens:[1,5],dimmedian_idx:5,direct:1,direct_attack:1,directori:[2,8],disabl:1,disconnect:1,disk:2,displai:8,distanc:8,distribut:[4,5,8],distributed_rank:8,doe:1,drop:[4,5],drop_edg:[4,5],drop_nod:[4,5],drop_path:[4,5],dropedg:[4,5],dropnod:[4,5],dropout1:5,dropout2:5,dropout:5,droppath:[4,5],dst:1,dtype:8,dure:[4,5,7],e:[1,7,8,9],each:[1,2,4,5],edg:[1,3,4,5,6,8],edge_flip:1,edge_index1:8,edge_index2:8,edge_index:[1,3,4,5,6,7,8],edge_weight:[1,3,4,5,8],edges_to_add:8,edges_to_flip:8,edges_to_remov:8,edisonleeee:[2,9],effect:1,ego_graph:8,eig_val:1,eig_vec:1,eigen:[1,3],eigendecomposit:3,eigenvalu:[5,7],eigenvector:[5,7],elast:5,elasticconv:5,elasticgnn:5,element:[1,3,4,8],elu:5,embed:[1,5],emp_forward:5,end:8,entri:8,environ:9,ep:[1,3,6],epoch:[1,7],eps_u:7,eps_v:7,epsilon:1,equal:8,equat:1,equival:[4,5],error:8,es:[1,3,6],estim:[6,8],estimate_self_training_label:6,etc:7,evalu:[3,7],everi:[1,2,3,5],exampl:[1,2,3,4,5,6,7,8],exclud:8,execut:5,exist:4,exp:8,expect:8,explain:1,expos:8,extend:5,extens:3,extern:1,extra:3,extra_repr:[3,7],extract:8,f:8,factor:3,fals:[1,3,4,5,6,8],fast:1,faster:8,feat:[1,8],feat_budget:1,feat_chang:1,feat_flip:1,feat_grad:1,feat_limit:1,feat_step:1,featur:[1,3,5,8,9],feature_attack:1,feature_scor:1,feel:9,fga:1,fgattack:1,fgbackdoor:1,fgsm:1,file:8,fill:4,fill_valu:4,filter:[1,3,8],filter_chisquar:8,first:5,fit:[3,7,8],five:5,flag:1,flatten:8,flip:[1,8],flip_attack:1,flip_edg:8,flipattack:1,floattensor:5,fly:5,follow:[1,2,7,8],forcibl:[1,3],form:[1,4,5],format:[2,8],format_num:8,former:[1,3,5],formul:1,forward:[1,3,5],framework:1,free:9,freez:[1,3,6],freeze_surrog:6,freezi:6,from:[1,2,3,4,5,6,7,8],func:8,further:5,g:[1,7],gamma:5,gat:5,gb:1,gcn:[3,5,7,8],gcnconv:5,gener:[1,8],get:[1,7,8],get_added_edg:1,get_attacker_nod:1,get_candidate_edg:1,get_candidate_featur:1,get_feat_perturb:1,get_feature_import:1,get_link_import:1,get_logg:8,get_perturbed_adj:1,get_perturbed_feat:1,get_removed_edg:1,get_subgraph:1,get_top_attack:1,get_train:7,gfa:1,gfattack:1,git:9,github:[2,5,8,9],given:8,glcore:5,gma:1,gnnguard:[3,5],grad_fn:5,gradient:[1,6],gradient_wrt_x:1,graph:[1,2,3,4,5,6,7,8],graphdata:2,graphwar:9,graphwardataset:[1,2],grb:8,guard:3,h1:5,h2:5,h3:5,h4:5,ha:[1,5],har:1,hat:5,have:8,head:[5,8],here:1,heurist:1,hid:[5,8],hidden:5,hide:1,high:1,histori:7,hook:[1,3,5],hop:8,how:[1,8],http:[2,5,8,9],hyperparamet:5,i:[1,5,8],iclr:[1,4,5,8],icml:[1,5],id:1,idx:8,ieee:1,ig:1,igattack:1,ignor:[1,3,5],ijcai:[1,3,5],implement:[1,3,5,8],importerror:[4,5],improv:5,in_channel:5,in_memory_dataset:2,inc_mat:5,incid:5,includ:[1,5,7,8],index:[0,1,4,5,8],indic:[1,5,8],indirect:1,individu:1,induc:8,inf:1,influenc:[1,3],influence_nod:1,info:8,inform:[1,3,6],inher:1,initi:[1,3,6,8],inject:0,inject_edg:1,inject_feat:1,inject_nod:1,injected_edg:1,injected_edge_index:1,injected_edge_weight:1,injected_feat:1,injected_nod:1,injection_attack:1,injectionattack:1,inmemorydataset:2,inner_train:1,input:[1,3,4,5,6,7,8],insert:8,insight:[1,3],instal:[4,5],instanc:[1,3,5,6,8],instead:[1,3,5],int64:8,integ:[4,5],intellig:8,interconnect:1,interfac:1,intermedi:[5,7],intern:1,interv:8,invalid:8,is_edge_index:6,is_legal_edg:1,is_singleton_edg:1,is_sort:4,issu:9,item:8,iter:[1,8],its:8,itself:[1,3,6,8],jaccard:3,jaccardpurif:3,jknet:5,jump:5,k:[1,3,5,8],k_hop_subgraph:8,kdd:[1,5,8],kei:8,kernel:8,keyword:[1,5,7],kl:7,knowledg:5,kwarg:[1,5,8],l1_project:5,l21:5,l21_project:5,l2:5,l638:8,l:5,label:[1,6,8],labeled_nod:1,lambda1:5,lambda2:5,lambda_:[1,5,7],lambda_amp:5,lambda_u:7,lambda_v:7,larg:1,larger:8,largest:[1,2,3,8],largestconnectedcompon:[1,2],last:8,latter:[1,3,5],law:8,layer:[0,4,8],lead:8,learn:[1,4,5],legal:1,length:[4,5,8],length_a:8,level:8,lgc_backdoor:1,lgcb:1,lgcbackdoor:1,li:1,like:[1,3,6,7,8],likelihood:8,likelihoodfilt:8,lim_max:8,lim_min:8,limit:[1,8],line:3,linear:8,link:1,list:[1,2,5,8],ll_constraint:1,ll_cutoff:[1,8],ll_ratio:8,loc:5,local:8,locat:[4,5],log:[7,8],logger:8,logit:1,longtensor:[4,5,6],loop:[3,5,8],loss:[1,7],low:[1,3],lr:[1,7],lstm:5,m1:8,m2:8,m:[4,5,6,8],mai:9,mainli:1,make:[1,8],mani:1,mask:[3,4,5,7,8],mask_or_not_given:7,maskga:[4,5],master:8,match:5,mathbf:5,matrix:[1,3,4,5,8],max:[1,5],max_perturb:1,maximum:[1,8],mayb:9,mean:8,median:5,medianconv:5,mediangcn:5,meet:[4,5,9],messag:[5,8],meta:1,metattack:1,method:[1,3,6,7],metric:8,min:1,minimum:8,minmax:1,minmaxattack:1,miss:8,mm:4,mode:[5,8],model1:8,model1_lay:8,model1_nam:8,model2:8,model2_lay:8,model2_nam:8,model:[0,1,3,4,6,7,8],modelcheckpoint:[3,7],modif:1,modifi:[1,5,8],modified_adj:1,modified_feat:1,modul:[0,1,3,5,6,7,8],momentum:1,monitor:[3,7],more:[2,5,8],msg:8,multi:3,multipl:[4,5],must:[1,5],my:8,my_ckpt:7,n:[1,5,8],n_influenc:1,n_old:8,n_per_class:8,name:[1,2,8],namedtupl:8,ndarrai:[1,8],need:[1,3,5],neighbor:[1,8],nettack:[1,2],network:[1,2,3,4,5,7,8],networkx:8,neural:[1,2,3,5,7,8],neurip:[3,4,5],new_n:8,new_s_d:8,newest:9,newli:1,nn:[0,1,3,4,6,7,8],node:[1,3,4,5,6,8],non:[1,4,8],none:[1,2,3,4,5,6,7,8],nonzero:1,norm:8,normal:[3,5,8],nosum:1,notat:8,note:1,noth:4,notimplementederror:1,now:5,np:[1,8],npz:2,num_attack:1,num_budget:1,num_class:[3,8],num_edges_glob:1,num_edges_loc:1,num_featur:3,num_head:5,num_nod:[1,4,5,8],numba:8,number:[1,3,4,5,6,8],numer:8,numpi:[1,8],object:[2,8],off:[5,7],old:8,onc:8,one:[1,2,3,5,8],ones:[5,8],onli:2,onto:5,open:9,oper:[1,5],optim:[1,7],option:[1,2,3,4,5,6,7,8],order:[1,3,8],org:[5,8],origin:1,other:[1,3,7],otherwis:[1,8],our:[5,9],out:[4,5,8],out_channel:5,outdat:9,output:[1,4,5,7,8],overlap:8,overrid:1,overridden:[1,3,5],own:3,p:[4,5],packag:9,page:9,pagerank:5,paper:[1,2,3,4,5,8],paramet:[1,2,3,4,5,6,7,8],pass:[1,3,5],patch:3,pbar:8,per:[4,5,8],percentag:[1,4,5,8],perform:[1,3,4,5],person:5,perspect:1,perturb:[1,3,7],perturbed_data:3,petarv:5,pgd:1,pgdattack:1,pick:7,pip:9,place:1,pleas:[1,5],plot_result:8,possibl:8,potenti:[1,8],power:8,powerlaw:8,pre_transform:2,predict:[1,5,7],predict_step:7,preserv:[5,8],principl:1,print:[3,8],probabl:[4,5],problem:9,progbar:[1,8],progress:8,project:[1,5],propag:5,properli:5,properti:[1,7],proximal_l21:5,pth:3,pubm:[1,2],purif:3,py:8,pyg:[1,3,6,7],pygat:5,pygdata:[1,2],pytorch:[5,8],q:[4,5],r:[4,5],radiu:[5,8],rais:[1,3,4,5,6,8],randn:[4,5],random:[1,3,4,5,6,8],random_inject:1,random_st:8,randomattack:1,randomguard:3,randominject:1,randomli:[1,8],rang:[4,5,8],rank:3,ratio:[1,5,8],re:3,recip:[1,3,5],recommend:[5,9],reconstruct:3,reduc:4,reduct:4,refer:5,regist:[1,3,5],regress:7,regression_loss:5,relu:[5,8],rememb:[1,8],remov:[1,3,8],remove_edg:[1,8],remove_edge_index:3,remove_feat:1,removed_edg:[1,3],removed_feat:1,repeat:8,repres:[1,3,8],represent:[3,5],reproduc:1,reproduct:6,requir:[1,3,5,6,8],reset:1,reset_paramet:[1,5],residu:5,respect:8,restrict:1,result:[1,3,4,5,7,8],robust:5,robustconv:5,robustgcn:[5,7],robustgcn_train:7,robustgcntrain:7,root:[1,2,4,5,8],row:5,row_norm:5,run:[1,3,5,8],runtimeerror:[1,3,6],runtimewarn:5,s:[1,3,5,6],s_d:8,s_old:8,same:[1,5,8],sampl:[1,4,5,8],sample_epoch:1,sat:[5,7],satconv:5,sattrain:7,save:[1,2,3,6,8],save_path:8,scalabl:1,scale:[1,3,5,7],scientif:8,scipi:[1,8],scipy_norm:8,scope:[4,5],score:[1,3],screen:8,second:8,see:[1,2,5,7],seed:[1,6,8],select:1,self:[1,3,5,8],semi:[5,8],sensit:7,separ:5,sequenc:8,sequenti:5,seri:2,set:[1,2,3,4,5,6,8],set_allow_feature_attack:1,set_allow_singleton:1,set_allow_structure_attack:1,set_max_perturb:1,set_norm:1,set_se:6,setup_logg:8,setup_surrog:[1,3,6],sever:1,sga:1,sgattack:1,sgc:5,sgconv:5,shape:[3,4,5,6,8],share:1,should:[1,2,3,5],silent:[1,3,5,8],similar:[3,5,8],simpgcn:[5,7],simpgcntrain:7,simpl:[1,5,7],simplifi:5,simultan:1,sinc:[1,3,5],singl:[1,2,3,8],singleton:[1,3,8],singleton_filt:8,singleton_mask:8,singletonfilt:8,singular:3,size:[1,3,5,8],smallest:8,social:1,soft:5,softmax:[1,3,6,7],softmedianconv:5,softmediangcn:5,solut:1,some:8,sort:[4,5],sort_edg:8,sourc:[1,2,3,4,5,6,7,8],sp:[1,8],spars:[1,4,5,8],sparse_matrix:8,sparsetensor:[3,4,5],specif:[1,7],specifi:[4,5,8],spectra:5,spectral:5,split:[3,8],split_nod:8,split_nodes_by_class:8,spmm:4,src:[1,8],ssgc:5,ssgconv:5,standard:[5,8],state:1,step:[1,5,7,8],store:2,str:[1,2,3,4,5,6,7,8],strategi:3,string:[3,8],strongest_wrong_class:1,structur:[1,4,5],structure_attack:1,structure_scor:1,sub_edg:1,sub_nod:1,subclass:[1,3,5],subgraph:[1,8],subgraph_process:1,sum:[1,4,8],supervis:[5,8],support:5,sure:1,surrog:[1,3,6],surrogate_model:1,surrogate_train:3,svd:3,svdpurif:3,symmetr:[1,3,5,8],t:[1,2,6],tagcn:5,tagcnconv:5,tagconv:5,take:[1,2,3,5],tanh:8,target:[0,3,8],target_class:1,target_label:1,target_nod:[1,3],targeted_attack:1,targetedattack:1,targets_class:1,teleport:5,temperatur:[1,3,6],tensor:[1,3,4,5,6,7,8],test:[7,8],test_mask:7,test_nod:[3,8],test_step:7,than:[5,8],them:[1,3,5],thi:[1,3,5,8],threshold:[1,3],through:5,thudm:8,time:[1,3,6],titl:8,tkde:1,to_dense_adj:4,to_sparse_tensor:4,to_tensor:8,top:[1,3,8],topk:8,topk_values_indic:8,topolog:[1,5],torch:[1,3,4,5,6,7,8],torch_clust:[4,5],torch_geometr:[1,2,3,5,8],torch_scatt:4,torch_spars:[3,4,5],total:[1,4,5,8],toward:[1,4,5],tqdm:1,trade:[5,7],train:[0,1,3,4,5,6,8],train_input:7,train_mask:7,train_nod:[3,8],train_step:7,trainer:[3,7],tran:1,transform:[1,2,7],trigger:1,tupl:[1,3,4,5,6,8],turn:8,two:[5,8],txt:8,type:[1,3,4,5,6,7,8],typeerror:[1,8],u:[1,5,8,9],understand:5,unexpect:1,uniform:[4,5],uniformli:[4,5],unimplemeted_model:7,union:[1,3,4,5,6,7,8],unit:5,unit_nam:8,univers:3,universal_defens:3,universaldefens:3,unlabeled_nod:1,untarget:0,untargeted_attack:1,untargetedattack:1,unus:5,updat:8,update_sx:8,us:[1,2,3,4,5,6,7,8],usag:5,usual:8,util:[0,1,5,7],v:[1,5,8],val:8,val_acc:[3,7],val_input:7,val_mask:7,val_nod:[3,8],valid:[7,8],valu:[1,3,4,5,8],value_for_last_step:8,valueerror:[4,5,8],varianc:5,variant:1,vector:1,verbos:[1,7,8,9],veri:5,version:[2,5],via:[1,9],victim_label:[1,3],victim_nod:1,virtual:9,visual:8,vulner:5,w:[1,8],wa:1,walk:[4,5],walk_length:[4,5],walks_per_nod:[4,5],warn:8,we:[1,2,3,5],weight:[1,4,5],weight_decai:7,when:8,where:[1,2,3,4,6,8],whether:[1,3,4,5,6,8],which:[1,2,3,4,5,6,8],whole:5,whose:1,width:8,wise:5,within:[1,3,5,8],without:5,work:[1,5],would:[1,5,6,8],wrap:8,wrapper:[5,8],wsdm:[5,8],x:[1,3,4,5,6,7],x_mean:1,xw:1,y:[3,7],you:[3,8,9],your:[1,3,7,8,9],your_mask:7,zero:[1,3,4]},titles:["Welcome to GraphWar\u2019s documentation!","graphwar.attack","graphwar.dataset","graphwar.defense","graphwar.functional","graphwar.nn","graphwar","graphwar.training","graphwar.utils","Installation"],titleterms:{"class":1,"function":4,api:0,attack:1,backdoor:1,base:1,dataset:2,defens:3,document:0,graphwar:[0,1,2,3,4,5,6,7,8],indic:0,inject:1,instal:[0,9],layer:5,model:5,nn:5,packag:0,refer:0,s:0,tabl:0,target:1,train:7,untarget:1,util:8,welcom:0}})