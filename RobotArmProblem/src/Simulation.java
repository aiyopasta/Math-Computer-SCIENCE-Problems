/*import lejos.hardware.*;
import lejos.remote.ev3.RMIRegulatedMotor;
import lejos.remote.ev3.RemoteEV3;
import lejos.robotics.RegulatedMotor;*/

import java.rmi.RemoteException;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.net.MalformedURLException;
import java.rmi.NotBoundException;
import java.util.*;

import javax.swing.JFrame;
import javax.swing.JPanel;

public class Simulation extends JPanel implements Runnable{

	JFrame frame = new JFrame();
	Thread t;

	ArrayList<Ellipse2D> points = new ArrayList<Ellipse2D>();

	final double BASE_LENGTH = 13; // Lego holes, iPad is 25 x 19 Lego holes.
	final int SCALE = 20; //Pixel:LegoHole ratio, don't set to 1 or it still adds some amount

	Rectangle2D iPad = new Rectangle2D.Double(500-((25*SCALE)/2), (SCALE*(int)BASE_LENGTH)-((19*SCALE)/2), 25*SCALE, 19*SCALE);

	Arm arm = new Arm(BASE_LENGTH);

	int k = -1;

//	RemoteEV3 ev3;
//	RMIRegulatedMotor extendMotor, rotateMotor;

	boolean bot = true;

	public Simulation() {

		try {
//			ev3 = new RemoteEV3("10.0.1.1");
		} catch (Exception e) {
			e.printStackTrace();
		}

		try {
//			rotateMotor = ev3.createRegulatedMotor("A", 'L');
//			extendMotor = ev3.createRegulatedMotor("B", 'M');
//
//			rotateMotor.setSpeed(80);
//			extendMotor.setSpeed(80);
		} catch(Exception ex){
			ex.printStackTrace();
			closePorts();
		}

		frame.setSize(1000, 1000);
		frame.setVisible(true);
		frame.add(this);
		frame.addWindowListener(new WindowAdapter() {

            @Override
            public void windowClosed(WindowEvent e) {}

            @Override
            public void windowClosing(WindowEvent e) {
                super.windowClosing(e);
	        	closePorts();
            }
        });
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		t = new Thread(this);
		t.start();
	}

	public void closePorts() {
		try {
//        	HelloWorld.this.rotateMotor.close();
//        	HelloWorld.this.extendMotor.close();
    	} catch (Exception ex){}

    	System.out.println("ports closed");
	}

	public double evalFunction(double x) {
		//ENTER FUNCTION HERE
		double y = Math.sin(x);
		return y;
	}

	public void paintComponent(Graphics g) {
		super.paintComponent(g);
		Graphics2D g2d = (Graphics2D)g;

		g2d.draw(iPad);

		g2d.setStroke(new BasicStroke(3));

		g2d.setColor(Color.BLACK);
		g2d.drawLine(500, 0, 500, SCALE * (int)BASE_LENGTH);

		g2d.setColor(Color.RED);


		int drawX = ((int)arm.getPoint().getX()) + (SCALE*Math.abs(500-((int)arm.getPoint().getX())))*(Integer.signum(((int)arm.getPoint().getX()) - 500));
		g2d.drawLine(500, 0, drawX, (SCALE*(int)arm.getPoint().getY()));
		points.add(new Ellipse2D.Double(drawX, (SCALE*arm.getPoint().getY()), 5, 5));

		g2d.setStroke(new BasicStroke(1));
		g2d.setColor(Color.BLUE);

		int p = 0;
		for (Ellipse2D point : points) {
			g2d.draw(point);
		}
	}

	@Override
	public void run() {

		for (int i=1;i>=-1;i-=2) {

			int drawX = 500;

			while (iPad.contains(new Point2D.Double(drawX, SCALE*arm.getPoint().getY())) && !arm.breakPoint) {

				drawX = ((int)arm.getPoint().getX()) + (SCALE*Math.abs(500-((int)arm.getPoint().getX())))*(Integer.signum(((int)arm.getPoint().getX()) - 500));

				moveArmTo(k+=i);

				repaint();
				revalidate();

				try {
					t.sleep(100); //see if code waits for physical motor to move. if it doesn't adjust sleep time accordingly.
				} catch(InterruptedException ex) {}
			}

			k-=2;
			moveArmTo(k);
		}

		System.out.println("Finished");
		closePorts();

	}

	public void moveArmTo(double xvalue) {
		double r = Math.sqrt(Math.pow(xvalue, 2) + Math.pow(evalFunction(xvalue), 2));
		double lo = BASE_LENGTH;
		double theta = Math.atan2(evalFunction(xvalue), xvalue);

		double newAngle = Math.asin((r*Math.sin(Math.toRadians(90)+theta))/(Math.sqrt(Math.pow(r, 2)+Math.pow(lo, 2)+2*lo*r*Math.sin(theta))));
		double lengthToMove = Math.sqrt(Math.pow(r, 2)+Math.pow(lo, 2)+2*lo*r*Math.sin(theta))-arm.length;

		double dy = newAngle - arm.angle;

		arm.moveArm(lengthToMove, newAngle);

		if (!arm.breakPoint && Math.abs(k)>0) {
			try {
				if (bot) {
//					rotateMotor.rotate((int)(5*Math.toDegrees(dy)));
//					extendMotor.rotate((int)-Math.toDegrees(lengthToMove/0.75));
				}
			} catch (Exception e) {
				e.getMessage();
				closePorts();
			}
		}
	}

	public class Arm {
		double length, angle = 0;
		boolean breakPoint = false;

		public Arm(double baseLength) {
			this.length = baseLength;
		}

		public void moveArm(double lengthToAdd, double newAngle) {

			if (length+lengthToAdd>=26) {
				breakPoint = true;
			} else {
				breakPoint = false;
				length+=lengthToAdd;
				angle=newAngle;
			}
		}

		public Point2D getPoint() {
			return new Point2D.Double(500-(length*Math.sin(angle)), length*Math.cos(angle));
		}

		public String toString() {
			return "Arm Length: " + length + " Angle From Vertical: " + Math.toDegrees(angle);
		}
	}

	public static double sin(double x) {return Math.sin(x);}
	public static double cos(double x) {return Math.cos(Math.toRadians(x));}
	public static double tan(double x) {return Math.tan(x);}
	public static double log(double x) {return Math.log(x);}
	public static double sqrt(double x) {return Math.sqrt(x);}
	public static double exp(double x) {return Math.pow(Math.E, x);}

	public static void main(String[] args) {
		Simulation app = new Simulation();
	}
}