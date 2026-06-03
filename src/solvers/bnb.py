from scipy.optimize import linprog
from collections import deque
import numpy as np
from copy import deepcopy, copy
from math import floor,ceil



# Node dictionary template

example_node = {
    "c" : None,
    "A_ub" : None,
    "b_ub" : None,
    "A_eq" : None,
    "b_eq" : None,
    "bounds" : None,
    "integer" : None,
}


def _normalize_linprog_duals(marginals, constraint_matrix):
    if constraint_matrix is None:
        return np.array([], dtype=float)

    duals = np.asarray(marginals, dtype=float)
    if duals.ndim == 0:
        n_constraints = constraint_matrix.shape[0]
        if n_constraints == 0:
            return np.array([], dtype=float)
        return duals.reshape(1)

    return duals.reshape(-1)






class BranchAndBound:
    """
        Implements a basic branch and bound algorithm, takes a dictionary defining the inital problem node as its input.
    
    """
    def __init__(self,init_node,sense):
        self.init_node = init_node
        self.integer = self.init_node["integer"]
        self.tree = []
        self.queue = deque()
        self.sol = None
        self.end_node = None
        self.pool = []


    def optimize_node(self,node):
        return linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )

# Handle infeasible sub sol
# Add parent and child relation

    def branch(self,node,x):
        # need to remove non integer
        # x = x[self.integer] 
        diff = np.abs(x) - np.floor(np.abs(x))
        diff = [val if isint else 0 for val,isint in zip(diff,self.integer)]
        # diff = [np.abs(xi - round(xi)) if isint else 0 for xi, isint in zip(x, self.integer)]
        if max(diff) < 1e-6:
            return None, None
        branch_var = np.argmax(diff)

        left_branch = deepcopy(node)
        right_branch = deepcopy(node)

        left_branch["bounds"][branch_var] = (left_branch["bounds"][branch_var][0],floor(x[branch_var]))
        right_branch["bounds"][branch_var] = (ceil(x[branch_var]),right_branch["bounds"][branch_var][1] )

        left_branch["parent"] = node
        right_branch["parent"] = node
        return left_branch,right_branch

        
    # def all_integer(self,x):
    #    return all( [not(var.is_integer() ^ bool(i)) for var,i in zip(x,self.integer)])
    def all_integer(self, x):
        return all((np.abs(var - round(var)) < 1e-6) if i else True for var, i in zip(x, self.integer))



    def solve(self,verbose = False):
        res = self.optimize_node(self.init_node)
        if not res.success:
            # print(res)
            # print("infeasible")
            return None
        self.tree.append(self.init_node)
        iter = 1
        if self.all_integer(res.x):
            # print(res.x[0].is_integer())
            self.sol = res
            self.init_node["sol"] = res
            self.end_node = self.init_node
            # print(f"Number of nodes explored: {iter}")


            return self.sol
        
        sol_rounded = np.floor(res.x) # Does this work for negative solutions?
        # sol_rounded = np.where(self.init_node["c"] >= 0,  np.ceil(res.x),np.floor(res.x))
        ub = self.init_node["c"] @ sol_rounded
        ub = float("inf")
        l,r = self.branch(self.init_node,res.x)
        self.queue.append(l)
        self.queue.append(r)
        self.init_node["children"].append(l)
        self.init_node["children"].append(r)
        while len(self.queue) > 0:
            iter += 1
            node = self.queue.popleft()
            self.tree.append(node)
            res = self.optimize_node(node)
            node["sol"] = res
            if verbose:
                print(node)

            if not res.success:
                print("infeasible")
                continue

            if self.all_integer(res.x):
                if res.fun <= ub:
                    ub = res.fun
                    self.sol = res
                    self.end_node = node
                    self.pool.append(res)

            elif res.fun <= ub:
                l,r = self.branch(node,res.x)

                self.queue.append(l)
                self.queue.append(r)
                node["children"].append(l)
                node["children"].append(r)
        if verbose:
            print(f"Number of nodes explored: {iter}")
        return self.sol





class BranchAndBoundRevamped:
    """
        Implements a basic branch and bound algorithm, takes a dictionary defining the inital problem node as its input.
    
    """
    def __init__(self,verbose = False):
        self.verbose = verbose

    def optimize_node(self,node):
        return linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )

# Handle infeasible sub sol
# Add parent and child relation

    def branch(self,bounds,x,integer):
        # need to remove non integer
        # x = x[self.integer] 
        diff = np.abs(x) - np.floor(np.abs(x))
        diff = [val if isint else 0 for val,isint in zip(diff,integer)]
        # diff = [np.abs(xi - round(xi)) if isint else 0 for xi, isint in zip(x, self.integer)]
        if max(diff) < 1e-6:
            return None, None
        branch_var = np.argmax(diff)

        left_branch = copy(bounds)
        right_branch = copy(bounds)

        left_branch[branch_var] =  (left_branch[branch_var][0],floor(x[branch_var]))
        right_branch[branch_var] = (ceil(x[branch_var]), right_branch[branch_var][1])

        return left_branch,right_branch,branch_var

        
    # def all_integer(self,x):
    #    return all( [not(var.is_integer() ^ bool(i)) for var,i in zip(x,self.integer)])
    def all_integer(self, x,integer):
        return all((np.abs(var - round(var)) < 1e-6) if i else True for var, i in zip(x,integer))



    def solve(self,init_node):
        verbose = self.verbose
        results = []
        queue = deque()
        
        
        res = self.optimize_node(init_node)
        if not res.success:
            # print(res)
            # print("infeasible")
            return None
        # self.tree.append(self.init_node)
        iter = 1
        integer  = init_node["integer"]
        
        if self.all_integer(res.x,integer):
   
            # print(f"Number of nodes explored: {iter}")

            results.append(  
                           
                {       
                    "fun" : res.fun,
                    "x" : res.x ,
                    "eqlin" : _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
                    "ineqlin" : _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
                    "lower" : res.lower.marginals,
                    "upper" : res.upper.marginals,
                    "fathomed" : False,
                    "node" : init_node,
                    "bounds" : init_node["bounds"]

                }
                
            )

            return results
        
        ub = float("inf")
        
        l,r,branch_var = self.branch(init_node["bounds"],res.x,integer)
        l_val = l[branch_var][1]
        r_val = r[branch_var][0]
        l_cond = [(branch_var,"<=",l_val)]
        r_cond = [(branch_var,">=",r_val)]
        queue.append((l,branch_var,l_cond))
        queue.append((r,branch_var,r_cond))
        
        while len(queue) > 0:
            iter += 1
            bounds,branch_var,conds = queue.popleft()
            node = {
                "c" : init_node["c"],
                "A_ub" : init_node["A_ub"],
                "b_ub" : init_node["b_ub"],
                "A_eq" : init_node["A_eq"],
                "b_eq" : init_node["b_eq"],
                "bounds" : bounds
            }
            res = self.optimize_node(node)
            
            if verbose:
                print(node)

            if not res.success:
                if verbose:
                    print("infeasible")
                continue

            if self.all_integer(res.x,integer):
                if res.fun <= ub:
                    ub = res.fun
                    results.append(  
                                    
                        {       
                            "fun" : res.fun,
                            "x" : res.x ,
                            "eqlin" : _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
                            "ineqlin" : _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
                            "lower" : res.lower.marginals,
                            "upper" : res.upper.marginals,
                            "fathomed" : False,
                            "conds" : conds,
                            "node" : node,
                            "bounds" : bounds
                        }
                        
                    )

            elif res.fun <= ub:
                l,r,branch_var = self.branch(bounds,res.x,integer)
                l_val = l[branch_var][1]
                r_val = r[branch_var][0]
                l_cond = conds + [(branch_var,"<=",l_val)]
                r_cond =  conds + [(branch_var,">=",r_val)]
                queue.append((l,branch_var,l_cond))
                queue.append((r,branch_var,r_cond))
            else:
                
                results.append(  
                                    
                    {       
                        "fun" : res.fun,
                        "x" : res.x ,
                        "eqlin" : _normalize_linprog_duals(res.eqlin.marginals, init_node["A_eq"]),
                        "ineqlin" : _normalize_linprog_duals(res.ineqlin.marginals, init_node["A_ub"]),
                        "lower" : res.lower.marginals,
                        "upper" : res.upper.marginals,
                        "fathomed" : True,
                        "conds" : conds,
                        "node" : node,
                        "bounds" : bounds

                        
                        
                    }
                        
                )
                
                
                
        if verbose:
            print(f"Number of nodes explored: {iter}")
        return results

