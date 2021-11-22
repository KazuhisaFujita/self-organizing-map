<!--
Since : 2020/12/08
Update: 2021/11/22
-->

# Self-organizing map and its alternatives

Self-organizing map (SOM) and its alternatives can quantize a dataset and extract topology as a network from the dataset.
These programs are Kohonen's SOM, neural gas (NG) (Martinetz, 1991), and growing neural gas (GNG) (Fritzke, 1994).
Kohonen's SOM is a typical SOM algorithm.
In kohonen_2d.py, the network of Kohonen's SOM is a two-dimensional lattice.
NG and GNG are alternatives of SOM.
The network of NG and GNG grows according to a dataset.

- som.py: Kohonen's SOM
- ng.py: Neural gas
- gng.py: Growing neural gas

## Requirement

My programs use Networkx for the manipulation of a networks.

## References

- Fritzke B (1994) A growing neural gas network learns topologies. In: Advances  in Neural Information Processing Systems 7, MIT Press, Cambridge, MA, USA,  NIPS'94, p 625–632.
- Martinetz T, Schulten K (1991) A "neural-gas" network learns topologies. Artificial Neural Networks I:397--402.
