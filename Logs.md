## distribution

**distribution**						**对应distribution**

没有transpose的gap+var loss



**distribution2**				

hard_margin_loss + average_margin_loss + var_loss



**distribution3**					

gap+ var_loss + average_margin_loss



**distribution4**				

gap_loss + var_loss*average_margin_loss



**distribution5**

首先关注ot、average、var后逐渐增大对gap的权重



**distribution6**						**对应distribution_v2**

首先关注ot、average、var后逐渐增大对gap和var的权重，总体上var一直存在，权重不变



**distribution7**						**(删除average的版本)对应distribution_v3**

gap(取平均)+var+average     效果不理想

gap(取平均)+var



**distribution8**			

gap+var







## MDGAT

#### r_mdgat			对应v1

欧式坐标编码key和query



#### r_mdgat2		对应v2

用极坐标编码value（内存爆炸问题）



#### r_mdgat3

用极坐标编码source desc再mlp成value

内存极其爆炸还有待优化



#### r_mdgat4

用极坐标直接编码query和key

