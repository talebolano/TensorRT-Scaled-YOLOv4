project(SortMot)

include_directories(
    ./
     )

set(SRC_LIST cholesky.cpp klmf.cpp lapjv.cpp track.cpp )
add_library(SortMot SHARED ${SRC_LIST} )
