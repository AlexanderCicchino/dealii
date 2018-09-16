// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------



// check if p4est does limit_level_difference_at_vertices in one 2d tree
// and in different trees
// test1 divides the lower-right cell of a square three times
// test2 does the same with a subdivided_hyper_cube

#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"
#include "coarse_grid_common.h"



template <int dim>
void
test(std::ostream & /*out*/)
{
  {
    parallel::distributed::Triangulation<dim> tr(MPI_COMM_WORLD);

    GridGenerator::hyper_cube(tr);
    tr.begin_active()->set_refine_flag();
    tr.execute_coarsening_and_refinement();
    tr.begin_active()->set_refine_flag();
    tr.execute_coarsening_and_refinement();
    tr.begin(1)->child(3)->set_refine_flag();
    tr.execute_coarsening_and_refinement();

    //    write_vtk(tr, "1");
    deallog << "cells test1: " << tr.n_active_cells() << std::endl;
  }
  {
    parallel::distributed::Triangulation<dim> tr(MPI_COMM_WORLD);

    GridGenerator::subdivided_hyper_cube(tr, 2);
    tr.begin_active()->set_refine_flag();
    tr.execute_coarsening_and_refinement();
    tr.begin(0)->child(3)->set_refine_flag();
    tr.execute_coarsening_and_refinement();

    //    write_vtk(tr, "2");
    deallog << "cells test2: " << tr.n_active_cells() << std::endl;
  }
}


int
main(int argc, char *argv[])
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  deallog.push("2d");
  test<2>(deallog.get_file_stream());
  deallog.pop();
}
