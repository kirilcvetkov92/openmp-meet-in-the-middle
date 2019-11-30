 #include <iostream>

#include <vector>

#include <algorithm>

#include <omp.h>

#include <fstream>

#include <time.h>

using namespace std;

void mergeB(vector < int > & b, int start, int enda, vector < int > & bNew) {
  int middle = (enda + start) / 2;

  int i = start, j = middle;
  int r = start;
  while (i != middle && j != enda)
    if (b[i] <= b[j])
      bNew[r++] = b[i++];
    else
      bNew[r++] = b[j++];

  while (i != middle)
    bNew[r++] = b[i++];

  while (j != enda)
    bNew[r++] = b[j++];

}

int numberOfWays(vector < int > x, int C) {

  double ctime = 0;
  int n1 = x.size() / 2, n2 = x.size() - x.size() / 2;
  vector < int > a(1 << n1), b(1 << n2);
  vector < int > bNew(1 << n2);
  vector < int > vNew(1 << n2);

  double startTime = omp_get_wtime();
  # pragma omp parallel for
  for (int i = 0; i < (1 << n1); i++) {
    a[i] = 0;

    for (int j = 0; j < n1; j++)
      if ((i >> j) & 1)
        a[i] += x[j];
  }
  ctime = omp_get_wtime() - startTime;
  cout << ctime << endl;

  startTime = omp_get_wtime();#
  # pragma omp parallel for
  for (int i = 0; i < (1 << n2); i++) {
    b[i] = 0;

    for (int j = 0; j < n2; j++)
      if ((i >> j) & 1)
        b[i] += x[n1 + j];
  }
  ctime = omp_get_wtime() - startTime;
  cout << ctime << endl;

  startTime = omp_get_wtime();#
  # pragma omp parallel sections {

    # pragma omp section
    sort(b.begin(), b.begin() + (1 << n2) / 4 - 1);#
    # pragma omp section
    sort(b.begin() + (1 << n2) / 4, b.begin() + 2 * (1 << n2) / 4 - 1);
    # pragma omp section
    sort(b.begin() + 2 * (1 << n2) / 4, b.begin() + 3 * (1 << n2) / 4 - 1);#
    # pragma omp section
    sort(b.begin() + 3 * (1 << n2) / 4, b.end());
  }

  
  # pragma omp parallel sections {
    
    # pragma omp section
    mergeB(b, 0, (1 << n2) / 2, bNew);
    # pragma omp section
    mergeB(b, (1 << n2) / 2, (1 << n2), bNew);
  }
  
  b.erase(b.begin(), b.end());
  mergeB(bNew, 0, (1 << n2), vNew);
  ctime = omp_get_wtime() - startTime;
  cout << ctime << endl;

  int ans = 0;
  startTime = omp_get_wtime();#
  # pragma omp parallel
  for reduction(+: ans)
  for (unsigned int i = 0; i < a.size(); i++) {
    int mi = 0, ma = vNew.size() - 1, av;
    while (mi < ma) {
      av = (mi + ma + 1) / 2;
      if (vNew[av] + a[i] <= C)
        mi = av;
      else
        ma = av - 1;
    }
    if (a[i] <= C)
      ans += mi + 1;
  }
  ctime = omp_get_wtime() - startTime;
  cout << ctime << endl;
  return ans;
}

int main() {
  ofstream myfile;

  for (int k = 1; k <= 4; k += 3) {
    omp_set_num_threads(k);
    int C, n;
    n = 55;
    C = 400;
    vector < int > x(n);
    double ctime = 0;
    double vreme = 0;
    for (int j = 0; j < 1; j++) {
      clock_t tStart = clock();#
      # pragma omp parallel for
      for (int i = 0; i < n; i++)
        x[i] = 200;

      double startTime = omp_get_wtime();
      cout << numberOfWays(x, C) << endl;
      //cout<<j<<endl;

      ctime += omp_get_wtime() - startTime;
      cout << ctime << endl;
    }

  }

  return 0;

}
