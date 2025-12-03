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
   cout << "\n\nhellllo\n\n";
   TList<int> mylist1;		// integer list
   cout <<  "\n" << mylist1.IsEmpty();
   TList<int> mylist5(5,3);
   PrintList(mylist5, "L5");
   cout << mylist1.IsEmpty()<<"\n\n";
   cout << mylist5.IsEmpty()<<"\n\n";
   mylist1.InsertFront(1);
   mylist1.InsertBack(4);
   PrintList(mylist1, "L1");

   mylist1.InsertBack(4);
   mylist1.InsertFront(2);

   cout << "---------------------------------------- \n" << endl;
   cout <<  "\n" << mylist1.IsEmpty();
   cout << "\n" << mylist1.GetSize();
   cout << "\n---------------------------------------- \n" << endl;
   PrintList(mylist1, "L1");
   cout << mylist1.GetFirst() << ": First\n\n";
   cout << mylist1.GetLast() << ": Last\n\n";
   

   mylist1.Clear();
   PrintList(mylist1, "L1");

    for (int i = 0; i < 10; i++){
      mylist1.InsertBack(i*2);
    }
    PrintList(mylist1, "L1");
	   
   TListIterator<int> itr = mylist1.GetIterator();
   mylist1.Insert(itr, 999);
   cout << "---------------------------------------- \n" ;
   cout <<  "\n" << mylist1.IsEmpty();
   cout << "\n---------------------------------------- \n";

   PrintList(mylist1, "L1");
   itr.Previous();	
   itr.Previous();	
   itr.Previous();	
   itr.Previous();	
   mylist1.Insert(itr, 999);

   PrintList(mylist1, "L1");

   TList<int> mylist2;
   for (int i = 0; i < 10; i++)
      mylist2.InsertBack(i * 3 + 1);

   PrintList(mylist2, "L2");

   //mylist1.Clear();

   //TList<int> mylist3 = mylist1 + TList<int>(100, 7);

   //TList<int> mylist4;
   //mylist4 = mylist2 + mylist1;

   //PrintList(mylist3, "L3");
   //PrintList(mylist4, "L4");

   //PrintList(mylist1, "hi");
   //mylist1.InsertFront(1);

   
   TList<int> myList;
    myList.InsertBack(1);
    myList.InsertBack(2);
    myList.InsertBack(3);

    // Create an iterator for the list
    TListIterator<int> iter = mylist1.GetIterator();

    // Use the iterator to iterate through the list

    PrintList(mylist1, "L1");
        // Access the data of the current node using GetData
        int currentValue = iter.GetData();
        cout << "Current Value: " << currentValue << endl;

        // Move to the next item in the list
        iter = iter.Next();
        currentValue = iter.GetData();
        cout << "Current Value: " << currentValue << endl;
        iter = iter.Next();
        currentValue = iter.GetData();
        cout << "Current Value: " << currentValue << endl;
        cout << "---------------------------------------- " << currentValue << endl;
        mylist1.IsEmpty();
        cout << "\n---------------------------------------- " << currentValue << endl;

   //PrintList(mylist1,"hello\n\n\n");
   //mylist1.IsEmpty();
   return 0;
   //TList mylist1;

   /*
   for (int i = 0; i < 10; i++)
	L1.InsertBack(i*2);

   PrintList(L1, "L1");

   for (int i = 0; i < 8; i++)
        L1.InsertFront( (i+1) * 100 );

   

   L1.RemoveBack();
   PrintList(L1, "L1");

   L1.RemoveBack();
   PrintList(L1, "L1");

   L1.RemoveFront();
   PrintList(L1, "L1");

   L1.RemoveFront();
   PrintList(L1, "L1");

   // try an iterator, and some middle inserts/deletes
   cout << "Testing some inserts with an iterator\n\n";

   TListIterator<int> itr = L1.GetIterator();
   L1.Insert(itr, 999);
   itr.Next();
   itr.Next();				// advance two spots
   L1.Insert(itr, 888);
   itr.Next();				
   itr.Next();				
   itr.Next();				// advance three spots
   L1.Insert(itr, 777);

   PrintList(L1, "L1");

   cout << "Testing some removes (with iterator)\n\n";

   itr.Next();   
   itr.Next();   			// advance two spots
   itr = L1.Remove(itr);		// delete current item
   PrintList(L1, "L1");

   for (int i = 0; i < 5; i++)
      itr.Previous();			// back 5 spots

   itr = L1.Remove(itr);		// delete current item
   PrintList(L1, "L1");
   
   // building another list

   TList<int> L2;
   for (int i = 0; i < 10; i++)
      L2.InsertBack(i * 3 + 1);

   PrintList(L2, "L2");

   // Testing + overload:
   TList<int> L3 = L1 + TList<int>(100, 7);

   TList<int> L4;
   L4 = L2 + L1;

   PrintList(L3, "L3");
   PrintList(L4, "L4");
   

*/



}