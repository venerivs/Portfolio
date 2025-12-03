
#include <iostream>
#include <string>

#include "tlist.h"

using namespace std;

template <typename T>
void PrintList(const TList<T>& L, string label)
{
   cout << label << " size is: " << L.GetSize() << "\n"
        << label << " = ";
   L.Print(cout);
   cout << "\n\n";
}

int main()
{
   TList<int> L1;		// integer list
   cout << "Add 9 values to L1 using a loop\n";
   for (int i = 0; i < 9; i++)
	L1.InsertBack(i*3);
   PrintList(L1, "L1");
   cout << "Add 1 value L1 using Insert Front\n";
   L1.InsertFront( 500 * 11 );
   PrintList(L1, "L1");

   cout << "Add 9 values to L1 using Insert Front\n";
   for (int i = 0; i < 9; i++)
        L1.InsertFront( (i+1) * 11 );
   PrintList(L1, "L1");
   cout << "Add 1 value to L1 using Insert Back\n";
   L1.InsertBack(777);
   PrintList(L1, "L1");

   cout << "Remove the last 2 values by using RemoveBack() twice\n";
   L1.RemoveBack();
   L1.RemoveBack();
   PrintList(L1, "L1");

   cout << "Remove the front 2 values by using RemoveFront() three times\n";
   L1.RemoveFront();
   L1.RemoveFront();
   PrintList(L1, "L1");

   cout << "Remove the last 3 values by using RemoveBack() twice\n";
   L1.RemoveBack();
   L1.RemoveBack();
   L1.RemoveBack();
   PrintList(L1, "L1");

   cout << "Remove the front 3 values by using RemoveFront() three times\n";
   L1.RemoveFront();
   L1.RemoveFront();
   L1.RemoveFront();
   PrintList(L1, "L1");

   cout << "Remove the last 2 values by using RemoveBack() twice\n";
   L1.RemoveBack();
   L1.RemoveBack();
   PrintList(L1, "L1");

   cout << "Remove the front 2 values by using RemoveFront() three times\n";
   L1.RemoveFront();
   L1.RemoveFront();
   PrintList(L1, "L1");

   cout << "Remove the last 3 values by using RemoveBack() twice\n";
   L1.RemoveBack();
   L1.RemoveBack();
   L1.RemoveBack();
   PrintList(L1, "L1");

   cout << "Remove the front 3 values by using RemoveFront() three times\n";
   L1.RemoveFront();
   L1.RemoveFront();
   L1.RemoveFront();
   PrintList(L1, "L1");

   cout << "Now lets create a new list using 5 Insert Iterators and 5 InsertBacks in a new L2\n";
   cout << "First 5 with InsertBack()\n";
   TList<int> L2;
   for (int i = 0; i < 5; i++)
      L2.InsertBack(i * 3 + 1);
   // try an iterator, and some middle inserts/deletes
   cout << "Testing some inserts with an iterator\n\n";

   TListIterator<int> itr = L2.GetIterator();
   TListIterator<int> itr2 = L2.GetIterator();
   cout << "Second 5 with Insert Iterator every other\n";
   L2.Insert(itr, 5555);
   itr = itr.Next();
   L2.Insert(itr, 4444);
   itr = itr.Next();
   L2.Insert(itr, 3333);
   itr = itr.Next();
   L2.Insert(itr, 2222);
   itr = itr.Next();
   L2.Insert(itr, 1111);
   PrintList(L2, "L2");
   cout << "After removing the third and the 6th from the list\n";
   for (int i = 0; i < 4; i++)
      itr = itr.Previous();
   itr2 = itr.Previous();
   L2.Remove(itr);
   itr = itr2;
   itr = itr.Previous();
   itr = itr.Previous();
   itr2 = itr.Previous();
   L2.Remove(itr);
   itr = itr2;
   PrintList(L2, "L2");


   cout << "Add on a second two at position three and six\n";
   itr = itr.Next();
   L2.Insert(itr, 999);
   for (int i = 0; i < 2; i++)
      itr = itr.Next();
   L2.Insert(itr, 888);
   PrintList(L2, "L2");

      cout << "After removing the second and third\n";
   for (int i = 0; i < 5; i++)
      itr = itr.Previous();
   itr2 = itr.Next();
   L2.Remove(itr);
   itr = itr2;
   itr2 = itr.Next();
   L2.Remove(itr);
   itr = itr2;
   PrintList(L2, "L2");
   cout << "Get data of itr\n";
   cout << itr2.GetData() << "\n";
   cout << "Does itr have a previous and next? 1 = yes, 0 = no\n";
   cout << itr2.HasNext() << " Next\n";
   cout << itr2.HasPrevious() << " Previous\n";

   cout << "Create L3\n";
   TList<int> L3;
   for (int i = 0; i < 9; i++)
	L3.InsertBack(i*3);
   PrintList(L3,"L3");
   

   // Testing + overload:
   cout<< "Testing the copy value function from List 3, by adding 7 different 100 values\n";
   PrintList(L3, "L3");
   TList<int> L4 = L3 + TList<int>(100, 7);
   PrintList(L4, "L4");
   cout << "Combining List 3 and L4 into L5\n";
   TList<int> L5;
   L5 = L3 + L4;
   PrintList(L5, "L5");



   cout << L5.GetFirst() << "<- First Value of L5\n" ;
   cout << L5.GetLast() << "<- Last Value of L5\n" ;



}
