from PEPit.function import Function


class SmoothHypoconvexPLFunction(Function):

    def __init__(self,
                 L,
                 m,
                 m_p,
                 is_leaf=True,
                 decomposition_dict=None,
                 reuse_gradient=False,
                 name=None):
        """

        Args:
            L (float): The quadratic upper bound parameter.
            is_leaf (bool): True if self is defined from scratch.
                            False if self is defined as linear combination of leaf.
            decomposition_dict (dict): decomposition of self as linear combination of leaf :class:`Function` objects.
                                       Keys are :class:`Function` objects and values are their associated coefficients.
            reuse_gradient (bool): If True, the same subgradient is returned
                                   when one requires it several times on the same :class:`Point`.
                                   If False, a new subgradient is computed each time one is required.
            name (str): name of the object. None by default. Can be updated later through the method `set_name`.

        """
        super().__init__(is_leaf=is_leaf,
                         decomposition_dict=decomposition_dict,
                         reuse_gradient=reuse_gradient,
                         name=name,
                         )

        # Store L
        self.L = L
        self.m = m
        self.m_p = m_p

    @staticmethod
    def set_convexity_constraint_i_j(xi, gi, fi,
                                     xj, gj, fj,
                                     ):
        """
        Formulates the list of interpolation constraints for self (CCP function).
        """
        # Interpolation conditions of convex functions class
        constraint = (fi - fj >= gj * (xi - xj))

        return constraint
    
    def set_hypoconvexity_constraint(self, xi, gi, fi,
                                     xj, gj, fj):
        constraint = (fi - fj >=
                      gj * (xi - xj)
                      + 1 / (2 * self.L) * (gi - gj) ** 2
                      + self.m / (2 * (1 - self.m / self.L)) * (
                              xi - xj - 1 / self.L * (gi - gj)) ** 2)

        return constraint

    
    def set_PL_constraint(self,
                                        xi, gi, fi,
                                        xj, gj, fj,
                                        ):
        constraint = (fi - fj - (1/(2 * self.m_p)) * gi ** 2 <= 0)

        return constraint

    def add_class_constraints(self):
        """
        Formulates the list of interpolation constraints for self (quadratically maximally growing convex function);
        see [1, Theorem 2.6].
        """
        if self.list_of_stationary_points == list():
            self.stationary_point()
        
        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_stationary_points,
                                                      constraint_name="PL",
                                                      set_class_constraint_i_j=self.set_PL_constraint,
                                                      )

        self.add_constraints_from_two_lists_of_points(list_of_points_1=self.list_of_points,
                                                      list_of_points_2=self.list_of_points,
                                                      constraint_name="hypoconvexity",
                                                      set_class_constraint_i_j=self.set_hypoconvexity_constraint,
                                                      )
