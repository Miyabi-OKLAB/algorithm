import java.util.Random

class cell
{
    private Random rand;
    private int threshold;
    private int id;

    cell(int id)
    {
        this.rand = new Random();
        this.threshold = 0;
        this.id = id;
    }

    public void setThreshold()
    {
        this.threshold = rand.nextInt(10) + 1;
    }

    public int getThreshold()
    {
        return this.threshold;
    }
}
