extern "C" {

struct Color {

        float r; //red
        float g; //green
        float b; //blue

        Color() { //default constructor
                r = 0;
                g = 0;
                b = 0;
        }

        Color(float red, float green, float blue) { //known value constructor
                r = red;
                g = green;
                b = blue;
        }
};

//void kernel16(int rank, Color* colArray);

void kernel2(int rank, Color* colArray);

}
